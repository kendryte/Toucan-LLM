from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import transformers
import sys
import llama_4bit as llama
import prompt
import copy
from PIL import Image

model_path="/data/xuchengzhen/alpaca/stanford_alpaca/llama7B-p50k-2/checkpoint-9252/"

tokenizer = LlamaTokenizer.from_pretrained(model_path)

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def show_img(path):
    return Image.open(path)


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
                                    use_cache=False,
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
    chatbot,
    temperature=0.5,
    top_p=0.75,
    max_new_tokens=1024,
    history=None,
    input=None,
    **kwargs,
):
    if history is None:
        history = []
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=40,
            num_beams=1,
            repetition_penalty=1.2,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    
    response = output.split("### Response:")[1].strip().replace("</s>","")
    query=instruction
    history.append([query, response]) 
       
    chatbot.append((parse_text(query), ""))
    chatbot[-1] = (parse_text(query), parse_text(response))       

    yield chatbot, history


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Toucan-4bit</h1>""")
#    gr.Image(value = show_img("../../resources/logo.png"),shape=(300, 300),type='pil',show_label=False,)

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=24, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            temperature = gr.Slider(0, 1, value=0.5, step=0.02, label="Temperature", interactive=True)
            top_p = gr.Slider(0, 1, value=0.75, step=0.01, label="Top P", interactive=True)
            max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Max tokens", interactive=True)
            emptyBtn = gr.Button("Clear History")

    state = gr.State([])
    submitBtn.click(evaluate, [user_input, chatbot, temperature, top_p,  max_new_tokens, state], [chatbot, state],show_progress=True)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot,state], show_progress=False)

demo.queue().launch(share=True, inbrowser=True)
