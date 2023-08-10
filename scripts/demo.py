import transformers
import gradio as gr
import mdtex2html
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import sys

model_path=""
tokenizer = LlamaTokenizer.from_pretrained(model_path)
LOAD_8BIT = False
BASE_MODEL = model_path

def show_img(path):
    return Image.open(path)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        meta_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request, in details and step by step."

        return f"""{meta_instruction}
        ### Instruction:{instruction}
        ### Response:"""

if not LOAD_8BIT:
    model.half()

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    chatbot,
    temperature=0.5,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.2,
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
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    
    response = output.split("### Response:")[1].strip().replace("</s>","")

    history.append([instruction, response]) 
       
    chatbot.append((parse_text(instruction), ""))
    chatbot[-1] = (parse_text(instruction), parse_text(response))       

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
    gr.HTML("""<h1 align="center">Toucan</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=24, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            temperature = gr.Slider(0, 1, value=0.5, step=0.01, label="Temperature", interactive=True)
            top_p = gr.Slider(0, 1, value=0.75, step=0.01, label="Top P", interactive=True)
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top K", interactive=True)
            num_beams = gr.Slider(1, 20, value=4, step=1, label="Beams", interactive=True)
            repetition_penalty = gr.Slider(1.0, 20.0, value=1.2, step=0.1, label="Repetition_penalty", interactive=True)
            max_new_tokens = gr.Slider(0, 1024, value=512, step=1.0, label="Max tokens", interactive=True)
            emptyBtn = gr.Button("Clear History")

    state = gr.State([])
    submitBtn.click(evaluate, [user_input, chatbot, temperature, top_p, top_k, num_beams,repetition_penalty, max_new_tokens, state], [chatbot, state],show_progress=False)

    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot,state], show_progress=False)

demo.queue().launch(share=True, inbrowser=True,)
