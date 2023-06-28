"""
Apply the delta weights on top of a base model.
Check if two modes share the same weights

Usage:
python3 check_delta.py --base ~/model_weights/llama-7b --target ~/model_weights/new-7b
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



def check_diff(base_model_path, target_model_path):
    print(f"Loading the target weights from {target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Check the difference")
    for name, param in tqdm(base.state_dict().items(), desc="Checking"):
        assert name in target.state_dict()
        if param.data.shape == target.state_dict()[name].shape:
            if torch.mean(param.data - target.state_dict()[name]) > 1e-5:
                print("Difference is large", name)
        else:
            print("Different shapes, this is unacceptable if you know what happened", name)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    args = parser.parse_args()

    check_diff(args.base_model_path, args.target_model_path)
