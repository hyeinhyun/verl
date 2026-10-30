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
"""Task-routed rewards for Vero-600K (https://huggingface.co/datasets/zlab-princeton/Vero-600k).

``examples/data_preprocess/vero.py`` prefixes ``data_source`` with ``vero/``. Every sample carries
``extra_info["reward_type"]`` which selects the verifier (numeric, multiple_choice, grounding, clicking,
instruction_following, ...). Responses must follow ``<think>...</think><answer>...</answer>``.
"""

from .reward import compute_score as _compute_score

# Weight of the format reward used by the Vero training recipe (gspo_llmjudge_shared.yaml).
VERO_FORMAT_SCORE = 0.2


def compute_score(solution_str, ground_truth, extra_info=None, format_score: float = VERO_FORMAT_SCORE):
    """Return ``{"score", "accuracy", "format"}`` for a Vero-600K sample.

    ``llm_judge`` samples always get ``accuracy=0``: they need a generative judge, which this
    rule-based reward does not run.
    """
    return _compute_score(solution_str, ground_truth, format_score=format_score, extra_info=extra_info)


__all__ = ["VERO_FORMAT_SCORE", "compute_score"]
