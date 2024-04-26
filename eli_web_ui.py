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

def generate_prompts(formatted_prompt, lines, temperature, top_p, num_generated, generated_lines_storage):      
    split_lines = (lines.split("\n"))
    your_line_idxs = [i for i in range(len(split_lines)) if len(split_lines[i].split(":")[1].strip()) == 0]

    idx = your_line_idxs[num_generated]
    
    try:
        next_line = split_lines[idx+1]
    except:
        next_line = "None"
    
    if num_generated == 0:
        gen_lines = []
    else:
        gen_lines = generated_lines_storage.split("<SPLIT>")
    
    for i in range(len(gen_lines)):
        split_lines[your_line_idxs[i]] += " " + gen_lines[i]

    prev_lines = np.array(split_lines)[list(range(0, idx+1))]
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
        temperature=temperature,
        top_p=top_p,
        model="claude-3-opus-20240229"
    )
        
    gen_line = message.content[0].text

    split_lines[your_line_idxs[num_generated]] += " " + gen_line
    gen_lines.append(gen_line)


    
    return "\n".join(np.array(split_lines)[list(range(0, idx+1))]), num_generated + 1, "<SPLIT>".join(gen_lines)
    
def load_saved():
    with open("./saved_prompts.json", "r") as f:
        saved = json.load(f)
    main_prompt = saved["main_prompt"] 
    scene_desc = saved["scene_desc"] 
    char_desc = saved["char_desc"] 
    lines = saved["lines"] 
    
    formatted_prompt = main_prompt

    formatted_prompt += f"\n\nScene Description:\n{scene_desc}\n"
    formatted_prompt += f"\nCharacter Description:\n{char_desc}\n"
    formatted_prompt += "\nLines:\n{lines}\n"
    formatted_prompt += "Next line:\n{next_line}\n"
    
    split_lines = (lines.split("\n"))
    num_to_generate = sum([1 for i in range(len(split_lines)) if len(split_lines[i].split(":")[1].strip()) == 0])
    
    info = f"""
    Number of lines to generate: {num_to_generate}
    """
    
    return formatted_prompt, lines, info

def reset():
    return "", 0, ""

with gr.Blocks(title="Adding Machine Prompt Editor", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("Edit prompt"):
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
                    save_btn = gr.Button(value="Save prompt", variant='primary')  
                    

        with gr.Tab("Generate Lines") as gen_lines_tab:
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.TextArea(label="Prompt template", interactive=False)
                with gr.Column():
                    gen_lines = gr.TextArea(label="Lines", interactive=False)
                    gen_info = gr.TextArea(label="Info", interactive=False)
                    
                    temperature = gr.Slider(label="Temperature (Higher = more creative, lower = more predictable response)", value=0.7, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    top_p = gr.Slider(label="Top P (Higher = larger set of words to choose from, lower = choose from the most likely words)", value=0.5, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    
            with gr.Row():
                generate_btn = gr.Button(value="Generate Next Line", variant="primary")
                reset_btn = gr.Button(value="Reset", variant="primary")
            with gr.Row():
                output = gr.TextArea(label="Generated Prompts", interactive=False)
                num_generated = gr.Number(value=0, visible=False)
                generated_lines_storage = gr.TextArea(interactive=False, value=None, visible=False)
                

    save_btn.click(fn=save_prompts, inputs=[main_prompt, scene_desc, char_desc, lines])
    gen_lines_tab.select(fn=load_saved, inputs=[], outputs=[gen_prompt, gen_lines, gen_info])
    generate_btn.click(fn=generate_prompts, inputs=[gen_prompt, gen_lines, temperature, top_p, num_generated, generated_lines_storage], outputs=[output, num_generated, generated_lines_storage])
    reset_btn.click(fn=reset, outputs=[output, num_generated, generated_lines_storage])
demo.launch()