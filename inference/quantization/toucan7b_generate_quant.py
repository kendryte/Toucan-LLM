import os
import sys
import torch
import transformers
import json
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from My_Quant import My_Quant, My_QuantModule, My_MatMul
import llama_4bit as llama



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


if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def validate_calib(calib_output_flag, calib_loader, calib_num, model):
    from tqdm import tqdm
    if calib_output_flag is True:
        output = []
    for i in tqdm(range(calib_num), desc="validate_calib"):
        prompt = generate_prompt(calib_loader[i], None)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=0.5,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            repetition_penalty=1.2,
        )
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1,
            )
        if calib_output_flag is True:
            output.append(out.to(device='cpu', non_blocking=True))
        torch.cuda.empty_cache()
    if calib_output_flag is True:
        return output

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
    # demo scripts for 4 bits weight quantization and fp16 activation
    # python toucan7b_generate_quant.py toucan7b_model_path  cal_dataset_json_path

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='toucan model to load')
    parser.add_argument('cal_dataset', type=str, help='json file for data calibration')
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--save_dir', type=str, default="./llama_4bit/pytorch_model_4bit.pt", help='quantized pt file')


    args = parser.parse_args()
    # testing code for readme
    res=[]
    instructions=[]
    with open(args.cal_dataset) as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        instructions.append(result['question'])

    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    LOAD_8BIT = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=LOAD_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

    if not LOAD_8BIT:
        model.half()  # seems to fix bugs for some users.

    model.eval()


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
    model_4bit = llama.LlamaForCausalLM(configuration)
    My_Quant(validate_calib=validate_calib,
                    calib_loader=instructions,
                    calib_num=8, # the number has to be smaller than the number of questions in test.json
                    model=model,
                    replace_op_model=model.model,
                    W_quant_method="GPTQ", #"GPTQ", 
                    A_quant_method="None", #"Uniform", #"None", 
                    W_MP_bit_list=[args.wbits],  
                    smooth_flag=False,
                    W_mp_method="R2", 
                    A_mp_method="R2", 
                    related_list=[['q_proj', 'k_proj', 'v_proj'], ['gate_proj', 'up_proj']],
                    )
    model_4bit.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data.float()
    model_4bit.model.norm.weight.data = model.model.norm.weight.data.float()
    model_4bit.lm_head.weight.data = model.lm_head.weight.data.float()
    for i in range(len(model_4bit.model.layers)):
        model_4bit.model.layers[i].input_layernorm.weight.data = model.model.layers[i].input_layernorm.weight.data.float()
        model_4bit.model.layers[i].post_attention_layernorm.weight.data = model.model.layers[i].post_attention_layernorm.weight.data.float()
        model_4bit.model.layers[i].self_attn.rotary_emb.inv_freq.data = model.model.layers[i].self_attn.rotary_emb.inv_freq.data.float()

        model_4bit.model.layers[i].self_attn.q_proj_weigth.data = model.model.layers[i].self_attn.q_proj.org_module.weight_int
        model_4bit.model.layers[i].self_attn.k_proj_weigth.data = model.model.layers[i].self_attn.k_proj.org_module.weight_int
        model_4bit.model.layers[i].self_attn.v_proj_weigth.data = model.model.layers[i].self_attn.v_proj.org_module.weight_int
        model_4bit.model.layers[i].self_attn.o_proj_weigth.data = model.model.layers[i].self_attn.o_proj.org_module.weight_int
        model_4bit.model.layers[i].self_attn.q_proj_delta_int.data = model.model.layers[i].self_attn.q_proj.org_module.delta_int
        model_4bit.model.layers[i].self_attn.k_proj_delta_int.data = model.model.layers[i].self_attn.k_proj.org_module.delta_int
        model_4bit.model.layers[i].self_attn.v_proj_delta_int.data = model.model.layers[i].self_attn.v_proj.org_module.delta_int
        model_4bit.model.layers[i].self_attn.o_proj_delta_int.data = model.model.layers[i].self_attn.o_proj.org_module.delta_int
        model_4bit.model.layers[i].self_attn.q_proj_s_delta.data = model.model.layers[i].self_attn.q_proj.org_module.s_delta.float()
        model_4bit.model.layers[i].self_attn.k_proj_s_delta.data = model.model.layers[i].self_attn.k_proj.org_module.s_delta.float()
        model_4bit.model.layers[i].self_attn.v_proj_s_delta.data = model.model.layers[i].self_attn.v_proj.org_module.s_delta.float()
        model_4bit.model.layers[i].self_attn.o_proj_s_delta.data = model.model.layers[i].self_attn.o_proj.org_module.s_delta.float()

        model_4bit.model.layers[i].mlp.gate_proj_weigth.data = model.model.layers[i].mlp.gate_proj.org_module.weight_int
        model_4bit.model.layers[i].mlp.down_proj_weigth.data = model.model.layers[i].mlp.down_proj.org_module.weight_int
        model_4bit.model.layers[i].mlp.up_proj_weigth.data = model.model.layers[i].mlp.up_proj.org_module.weight_int
        model_4bit.model.layers[i].mlp.gate_proj_delta_int.data = model.model.layers[i].mlp.gate_proj.org_module.delta_int
        model_4bit.model.layers[i].mlp.down_proj_delta_int.data = model.model.layers[i].mlp.down_proj.org_module.delta_int
        model_4bit.model.layers[i].mlp.up_proj_delta_int.data = model.model.layers[i].mlp.up_proj.org_module.delta_int
        model_4bit.model.layers[i].mlp.gate_proj_s_delta.data = model.model.layers[i].mlp.gate_proj.org_module.s_delta.float()
        model_4bit.model.layers[i].mlp.down_proj_s_delta.data = model.model.layers[i].mlp.down_proj.org_module.s_delta.float()
        model_4bit.model.layers[i].mlp.up_proj_s_delta.data = model.model.layers[i].mlp.up_proj.org_module.s_delta.float()
    torch.save(model_4bit.state_dict(), args.save_dir)

