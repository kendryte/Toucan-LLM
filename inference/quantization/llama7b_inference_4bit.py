import os
import sys
import torch
import transformers
import json
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import llama_4bit as llama
from huggingface_hub import snapshot_download


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""



def evaluate(
    instruction,
    input=None,
    temperature=0.5,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.2,
    max_new_tokens=1024,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    # testing code for readme
    # python llama7b_inference_4bit.py llama_model_save_path

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to save')

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained("kendryte/Toucan-llm-4bit", trust_remote_code=True)

    snapshot_download(repo_id="kendryte/Toucan-llm-4bit", 
                        local_dir=args.model,
                        local_dir_use_symlinks=False)
    model_path = os.path.join(args.model,"pytorch_model_4bit.pt")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    configuration = llama.LlamaConfig(vocab_size=49954,
                                        hidden_size=4096,
                                        intermediate_size=11008,
                                        num_hidden_layers=32,
                                        num_attention_heads=32,
                                        hidden_act="silu",
                                        max_position_embeddings=2048,
                                        initializer_range=0.02,
                                        rms_norm_eps=1e-6,
                                        use_cache=True,
                                        pad_token_id=0,
                                        bos_token_id=1,
                                        eos_token_id=2,
                                        tie_word_embeddings=False)

    model = llama.LlamaForCausalLM(configuration)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device=torch.device(device))

    model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    while True:
        instruction = input("请输入 instruction:")
        Response=evaluate(instruction).replace("</s>","")
        print("Response:",Response)

