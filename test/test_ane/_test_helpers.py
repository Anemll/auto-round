# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def model_infer(model, tokenizer, apply_chat_template=False):
    prompts = [
        "Hello,my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    if apply_chat_template:
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        prompts = texts

    inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        do_sample=False,  ## change this to follow official usage
        max_new_tokens=5,
    )
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {decoded_outputs[i]}")
        print("-" * 50)
    return decoded_outputs[0]

