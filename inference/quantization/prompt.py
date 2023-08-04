import transformers
import copy

class prompt:
    def __init__(self, tokenizer, max_len=2048, add_eos=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_eos=add_eos

class instruct_prompt(prompt):
    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    prompt_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Input:{input}\n\n### Response:"
    )
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def preprocess_gen(self, data_point):
        if 'history' not in data_point:
            if 'input' in data_point:
                user_prompt = self.prompt_input.format_map(data_point)
            else:
                user_prompt = self.prompt.format_map(data_point)
        else:
            user_prompt = "\n".join([i['input']+"\n"+i['output'] for i in data_point['history']]) + "\n" + data_point['input'] + "\n"
            user_prompt = user_prompt.replace("\n\n","\n")
            user_prompt = user_prompt[-self.max_len:]
        user_prompt=self.prompt.format_map({'instruction':user_prompt})
        input_ids = self.tokenizer(user_prompt)["input_ids"]
        return input_ids

    def postprocess(self, text, render=True):
        output = text.split("### Response:")[1].strip()
        output = output.replace("Belle", "Toucan")
        if '###' in output:
            output = output.split("###")[0]
        if 'User' in output:
            output = output.split("User")[0]
        output = output.replace('�','').replace('</s>', '') 
        if render:
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

