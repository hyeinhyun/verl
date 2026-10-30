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
"""
Preprocess Vero-600K (https://huggingface.co/datasets/zlab-princeton/Vero-600k) to parquet format.

Each subset is written to ``<local_save_dir>/<subset>/{train,val}.parquet``. ``data_source`` is prefixed with
``vero/`` so that ``default_compute_score`` scores it with ``verl.utils.reward_score.vero``, which routes on
``extra_info["reward_type"]``.

Samples with ``reward_type == "llm_judge"`` are dropped: they need a generative judge, so the rule-based reward
would only score their format.

Example:
    python examples/data_preprocess/vero.py --subsets stem-geo170k chart_ocr-chartqa_difficulty \
        --normalize_bbox_to_1000
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "zlab-princeton/Vero-600k"

# Subsets made only of llm_judge samples; skipped by default to avoid downloading them (~220 GB).
LLM_JUDGE_SUBSETS = {
    "captioning_IF-flickr30k",
    "captioning_IF-pixmo_ask_model_anything",
    "captioning_IF-pixmo_cap",
    "captioning_IF-pixmo_cap_qa_images",
}

# System prompt of the Vero recipe (vero-rl/examples/prompts/system_prompt_chatting.txt). The reward requires
# "<think>...</think><answer>...</answer>" and a single \boxed{} for answers with a definitive result.
SYSTEM_PROMPT = r"""You are a helpful, conversational assistant tasked with answering a question about an image.

Your response must include two parts:

1. **Reasoning**: A detailed, free-flowing chain of thought enclosed in `<think>` and `</think>` tags.
2. **Final Answer**: A clear, conversational response enclosed in `<answer>` and `</answer>` tags, using \boxed{} notation when the question has a definitive answer.

---

### Reasoning Instructions

* The reasoning section must be inside `<think>` … `</think>` tags.
* The reasoning should resemble a stream of consciousness: explore, test hypotheses, backtrack if necessary, reflect, and refine.
* Let the reasoning flow naturally while progressing toward a conclusion.
* Use reasoning strategies such as:
  * **Planning** – outline possible approaches before committing.
  * **Exploration** – consider multiple image regions or interpretations, even unlikely ones.
  * **Evaluation** – compare alternatives and verify against visual evidence.
  * **Reflection** – revisit earlier ideas if they may still be viable.
* Thoroughly examine and cross-check relevant image regions before narrowing down.
* If the image is ambiguous, make a reasonable inference based on visual and contextual cues.
* End the reasoning once you are confident in the conclusion.

---

### Final Answer Instructions

* The answer section must be enclosed in `<answer>` … `</answer>` tags.
* The `<answer>` section should stand on its own as a response to the user: it must provide necessary context and justification so that a reader can understand and verify the conclusion without reading `<think>`.
  - Do NOT refer to the `<think>` section (avoid phrases like “as explained above” or “from the reasoning”).
* Boxed result:
    * If the question has a definitive, concise answer (a number, word, phrase, or label), include a conversational, natural response followed by exactly one boxed result using LaTeX: \boxed{final_result}.
    * If the question is open-ended, subjective, or does not yield a concise final result, omit the boxed notation.

---

### Format Example

```
<think>
Detailed reasoning goes here...
</think>
<answer>
Self-contained response goes here...
Following the response, if a concise final result exists, include: \boxed{final_result}. If open-ended or no concise result, respond naturally without \boxed.
</answer>
```"""  # noqa: E501


def make_map_fn(normalize_bbox_to_1000: bool):
    def process_fn(example):
        prompt = example.pop("prompt")
        if isinstance(prompt, dict):  # HF stores a sequence of structs column-wise
            prompt = [{"role": r, "content": c} for r, c in zip(prompt["role"], prompt["content"], strict=True)]
        extra_info = dict(example.pop("extra_info"))
        if not any(m["role"] == "system" for m in prompt):
            prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt

        image = example.pop("image")
        # Grounding/clicking rewards need the original image size to interpret (or normalize) coordinates.
        extra_info["image_sizes"] = [[image.width, image.height]]
        extra_info["normalize_bbox_to_1000"] = normalize_bbox_to_1000

        return {
            "data_source": f"vero/{example['data_source']}",
            "prompt": prompt,
            "images": [image],
            "ability": example["ability"],
            "reward_model": example["reward_model"],
            "extra_info": extra_info,
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/vero", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--subsets", nargs="+", default=None, help="Subsets (HF configs) to convert, e.g. stem-geo170k. Default: all."
    )
    parser.add_argument(
        "--normalize_bbox_to_1000",
        action="store_true",
        help="Score grounding/clicking in 0-1000 normalized coordinates (Qwen3-VL). Default: absolute pixels.",
    )
    parser.add_argument("--num_proc", type=int, default=8)

    args = parser.parse_args()
    dataset_path = args.local_dataset_path or DATA_SOURCE
    subsets = args.subsets or [
        subset for subset in datasets.get_dataset_config_names(dataset_path) if subset not in LLM_JUDGE_SUBSETS
    ]
    local_save_dir = os.path.expanduser(args.local_save_dir)

    for subset in subsets:
        dataset = datasets.load_dataset(dataset_path, subset)
        for split, split_dataset in dataset.items():
            split_dataset = split_dataset.filter(
                lambda extra_info: (extra_info.get("reward_type") or "").strip().lower() != "llm_judge",
                input_columns="extra_info",
                num_proc=args.num_proc,
            )
            if len(split_dataset) == 0:
                print(f"Skipping {subset}/{split}: only llm_judge samples")
                continue
            split_dataset = split_dataset.map(
                function=make_map_fn(args.normalize_bbox_to_1000),
                remove_columns=split_dataset.column_names,
                num_proc=args.num_proc,
            )
            split_dataset.to_parquet(os.path.join(local_save_dir, subset, f"{split}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
