"""

Changed from https://github.com/lm-sys/FastChat/blob/main/fastchat/model/apply_delta.py
Apache License 2.0 https://github.com/lm-sys/FastChat/blob/main/LICENSE

Apply the delta weights on top of a base model.

Usage:

python scripts/apply_delta.py  --base /path_to_llama/llama-7b-hf --target ./delta_model/toucan-7b  --delta ./delta_model/toucan-7b-delta/

"""
import argparse
import gc
import glob
import json
import os
import shutil
import tempfile

from huggingface_hub import snapshot_download
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


GB = 1 << 30


def split_files(model_path, tmp_path, split_size):
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    file_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(file_pattern)

    part = 0
    try:
        for file_path in tqdm(files):
            state_dict = torch.load(file_path)
            new_state_dict = {}

            current_size = 0
            for name, param in state_dict.items():
                param_size = param.numel() * param.element_size()

                if current_size + param_size > split_size:
                    new_file_name = f"pytorch_model-{part}.bin"
                    new_file_path = os.path.join(tmp_path, new_file_name)
                    torch.save(new_state_dict, new_file_path)
                    current_size = 0
                    new_state_dict = None
                    gc.collect()
                    new_state_dict = {}
                    part += 1

                new_state_dict[name] = param
                current_size += param_size

            new_file_name = f"pytorch_model-{part}.bin"
            new_file_path = os.path.join(tmp_path, new_file_name)
            torch.save(new_state_dict, new_file_path)
            new_state_dict = None
            gc.collect()
            new_state_dict = {}
            part += 1
    except Exception as e:
        print(f"An error occurred during split_files: {e}")
        shutil.rmtree(tmp_path)
        raise




def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the delta weights from {delta_path}")
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Applying the delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        assert name in base.state_dict()
        if param.data.shape == base.state_dict()[name].shape:
            param.data += base.state_dict()[name]
        else:
            print(name, "is not the same shape as base model, use delta weights directly")
            import ipdb; ipdb.set_trace()


    print(f"Saving the target model to {target_model_path}")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
