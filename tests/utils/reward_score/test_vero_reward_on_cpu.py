# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

pytest.importorskip("math_verify")

from verl.utils.reward_score import default_compute_score  # noqa: E402
from verl.utils.reward_score.vero import VERO_FORMAT_SCORE, compute_score  # noqa: E402


def _response(answer: str, think: str = "Looking at the image.") -> str:
    return f"<think>{think}</think>\n<answer>{answer}</answer>"


def _score(answer, ground_truth, reward_type, **extra_info):
    return compute_score(_response(answer), ground_truth, extra_info={"reward_type": reward_type, **extra_info})


@pytest.mark.parametrize(
    "answer, ground_truth, reward_type, extra_info, expected_accuracy",
    [
        (r"The value is \boxed{6.7}.", "6.67", "numeric", {"tolerance": 0.05}, 1.0),
        (r"The value is \boxed{8}.", "6.67", "numeric", {"tolerance": 0.05}, 0.0),
        (r"The answer is \boxed{B}.", "B", "multiple_choice", {}, 1.0),
        (r"The answer is \boxed{C}.", "B", "multiple_choice", {}, 0.0),
        (r"It is a \boxed{red apple}.", "red apple", "string_match", {}, 1.0),
        (r"There are \boxed{3} dogs.", "3", "counting", {}, 1.0),
        (r"There are \boxed{4} dogs.", "3", "counting", {}, 0.0),
    ],
)
def test_boxed_reward_types(answer, ground_truth, reward_type, extra_info, expected_accuracy):
    result = _score(answer, ground_truth, reward_type, **extra_info)
    assert result["format"] == 1.0
    assert result["accuracy"] == expected_accuracy
    assert result["score"] == pytest.approx(
        (1 - VERO_FORMAT_SCORE) * expected_accuracy + VERO_FORMAT_SCORE * result["format"]
    )


def test_missing_boxed_halves_format_score():
    result = _score("The answer is B.", "B", "multiple_choice")
    assert result["format"] == 0.5


def test_bad_format_zeroes_reward():
    result = compute_score(r"The answer is \boxed{B}.", "B", extra_info={"reward_type": "multiple_choice"})
    assert result == {"score": 0.0, "accuracy": 0.0, "format": 0.0}


def test_empty_think_zeroes_reward():
    result = compute_score(
        r"<think> </think><answer>\boxed{B}</answer>", "B", extra_info={"reward_type": "multiple_choice"}
    )
    assert result["format"] == 0.0


@pytest.mark.parametrize(
    "predicted_box, normalize_bbox_to_1000, expected_accuracy",
    [
        ([100, 50, 300, 150], False, 1.0),
        ([500, 400, 600, 480], False, 0.0),
        # Image is 1000x500, so the absolute GT box maps to [100, 100, 300, 300] in 0-1000 space.
        ([100, 100, 300, 300], True, 1.0),
    ],
)
def test_grounding(predicted_box, normalize_bbox_to_1000, expected_accuracy):
    pytest.importorskip("scipy")
    ground_truth = json.dumps([{"bbox_2d": [100, 50, 300, 150]}])
    answer = json.dumps([{"bbox_2d": predicted_box}])
    result = _score(
        answer,
        ground_truth,
        "grounding",
        image_sizes=[[1000, 500]],
        normalize_bbox_to_1000=normalize_bbox_to_1000,
    )
    assert result["format"] == 1.0
    assert result["accuracy"] == pytest.approx(expected_accuracy)


@pytest.mark.parametrize(
    "point, normalize_bbox_to_1000, expected_accuracy",
    [
        ([200, 100], False, 1.0),
        ([800, 400], False, 0.0),
        ([200, 200], True, 1.0),
    ],
)
def test_clicking(point, normalize_bbox_to_1000, expected_accuracy):
    ground_truth = json.dumps([{"bbox_2d": [100, 50, 300, 150]}])
    answer = json.dumps([{"point_2d": point}])
    result = _score(
        answer, ground_truth, "clicking", image_sizes=[[1000, 500]], normalize_bbox_to_1000=normalize_bbox_to_1000
    )
    assert result["accuracy"] == expected_accuracy


def test_clicking_without_image_size_scores_zero():
    ground_truth = json.dumps([{"bbox_2d": [100, 50, 300, 150]}])
    result = _score(json.dumps([{"point_2d": [200, 100]}]), ground_truth, "clicking")
    assert result["accuracy"] == 0.0


def test_instruction_following():
    pytest.importorskip("langdetect")
    pytest.importorskip("nltk")
    ground_truth = json.dumps(
        {
            "instructions": [
                {"class": "LowercaseLettersEnglishChecker", "args": {}},
                {"class": "ForbiddenWords", "args": {"forbidden_words": ["cat"]}},
            ]
        }
    )
    assert _score("a small dog sits on the grass.", ground_truth, "instruction_following")["accuracy"] == 1.0
    assert _score("A small cat sits on the grass.", ground_truth, "instruction_following")["accuracy"] == 0.0


def test_llm_judge_gets_format_reward_only():
    result = _score("A white bird stands in the water.", "A bird in a lake.", "llm_judge")
    assert result["accuracy"] == 0.0
    assert result["format"] == 1.0


def test_default_compute_score_routes_vero_prefix():
    result = default_compute_score(
        "vero/CoSyn_400k_chart",
        _response(r"\boxed{6.67}"),
        "6.67",
        extra_info={"reward_type": "numeric", "tolerance": 0.05},
    )
    assert result["accuracy"] == 1.0
