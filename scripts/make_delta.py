"""
Changed from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/make_delta.py
Apache License 2.0 https://github.com/lm-sys/FastChat/blob/main/LICENSE

Make the delta weights by subtracting base weights.

Usage:
python3 make_delta.py --base ~/model_weights/llama-7b --target ~/model_weights/toucan-7b --delta ~/model_weights/toucan-7b-delta 
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def make_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the target model from {target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, use_fast=False)

    print("Calculating the delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        assert name in base.state_dict()
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        else:
            print(name, "not the same shape as base model")
            param.data = base.state_dict()[name]

    print(f"Saving the delta to {delta_path}")
    target.save_pretrained(delta_path)
    target_tokenizer.save_pretrained(delta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path)
