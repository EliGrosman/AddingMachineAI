import gradio as gr
import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = Anthropic(api_key = os.environ["ANTHROPIC_API_KEY"])

if os.path.isfile("./saved_prompts.json"):
    fname = "saved_prompts.json"
else:
    fname = "text_templates.json"

with open(fname, "r") as f:
    templates = json.load(f)

def save_prompts(main_prompt, scene_desc, char_desc, lines):
    templates["main_prompt"] = main_prompt
    templates["scene_desc"] = scene_desc
    templates["char_desc"] = char_desc
    templates["lines"] = lines

    with open("./saved_prompts.json", "w") as f:
        json.dump(templates, f)

def generate_prompts(main_prompt, scene_desc, char_desc, lines):
    formatted_prompt = main_prompt

    formatted_prompt += f"\n\nScene Description:\n{scene_desc}"
    formatted_prompt += f"\n\Character Description:\n{char_desc}"
    formatted_prompt += "\n\Lines:\n{lines}"
    formatted_prompt += "Next line:\n{next_line}"

    split_lines = (lines.split("\n"))
    your_line_idxs = [i for i in range(len(split_lines)) if len(split_lines[i].split(":")[1].strip()) == 0]

    for i in range(len(your_line_idxs)):
        idx = your_line_idxs[i]
        prev_lines = np.array(split_lines)[list(range(0, idx+1))]
        try:
            next_line = split_lines[idx+1]
        except:
            next_line = "None"

        lines_template = "\n".join(prev_lines)
        p = formatted_prompt.format(lines=lines_template, next_line=next_line)

        message = client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": p
                }
            ],
            temperature=0.9,
            top_p=0.5,
            model="claude-3-opus-20240229"
        )

        split_lines[your_line_idxs[i]] += message.content[0].text
    print(split_lines)
    


with gr.Blocks(title="Adding Machine Prompt Editor", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        main_prompt = gr.TextArea(label="Prompt template", value=templates["main_prompt"], interactive=True)

    with gr.Row():
        with gr.Column():
            scene_desc = gr.TextArea(label="Scene Description", value=templates["scene_desc"], interactive=True)
        with gr.Column():
            char_desc = gr.TextArea(label="Character Description", value=templates["char_desc"], interactive=True)
    with gr.Row():
        lines = gr.TextArea(label="Lines", value=templates["lines"], interactive=True)

    with gr.Row():
        with gr.Column():
            save_btn = gr.Button(value="Save prompt", variant='primary')  
        with gr.Column():
            generate_btn = gr.Button(value="Generate lines", variant='primary')

    save_btn.click(fn=save_prompts, inputs=[main_prompt, scene_desc, char_desc, lines])
    generate_btn.click(fn=generate_prompts, inputs=[main_prompt, scene_desc, char_desc, lines])

demo.launch()