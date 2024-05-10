import gradio as gr
import pandas as pd
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from resemble_wrapper import Resemble_Wrapper
import re
from scipy.io import wavfile
import uuid

load_dotenv()

client = Anthropic(api_key = os.environ["ANTHROPIC_API_KEY"])
resemble = Resemble_Wrapper(os.environ["RESEMBLE_API_KEY"])

if os.path.isfile("./saved_prompts.csv"):
    prompts = pd.read_csv("./saved_prompts.csv")
else:
    prompts = pd.DataFrame({"prompt_name": [], "main_prompt": [], "scene_desc": [], "char_desc": [], "lines": []})
    prompts.to_csv("./saved_prompts.csv", index = False)

def select_prompt(choice):
    global prompts
    if type(choice) is not type([]):
        row = prompts[prompts["prompt_name"] == choice].iloc[0]
        return row["prompt_name"], row["main_prompt"], row["scene_desc"], row["char_desc"], row["lines"]
    return "", "", "", "", ""

def select_prompt2(choice):
    global prompts
    if type(choice) is not type([]):
        row = prompts[prompts["prompt_name"] == choice].iloc[0]
        
        formatted_prompt = row["main_prompt"] + "\n\nScene Description:\n" + \
                           row["scene_desc"] + "\n\nCharacter Description:\n" + \
                           row["char_desc"] + "\n\nLines:\n" + \
                           "{lines}"
        return formatted_prompt, row["lines"]
    return "", ""


def overwrite_prompt(prompt_name, main_prompt, scene_desc, char_desc, lines):
    global prompts
    prompts.loc[prompts["prompt_name"] == prompt_name, "main_prompt"] = main_prompt
    prompts.loc[prompts["scene_desc"] == scene_desc, "scene_desc"] = scene_desc
    prompts.loc[prompts["char_desc"] == char_desc, "char_desc"] = char_desc
    prompts.loc[prompts["lines"] == lines, "lines"] = lines
    
    prompts.to_csv("./saved_prompts.csv", index = False)
    
    gr.Info(f"Overwrote prompt {prompt_name}")
    return gr.Dropdown(choices = list(prompts["prompt_name"]))
    
def save_new(prompt_name, main_prompt, scene_desc, char_desc, lines):
    global prompts
    if prompt_name in list(prompts["prompt_name"]):
        gr.Warning(f"Prompt '{main_prompt}' already exists")
    else:
        new_r = pd.DataFrame({"prompt_name": [prompt_name], "main_prompt": [main_prompt], "scene_desc": [scene_desc], "char_desc": [char_desc], "lines": [lines]})
        prompts = pd.concat([prompts, new_r])
        prompts.to_csv("./saved_prompts.csv", index = False)
        
        gr.Info(f"Saved prompt {prompt_name}")
    return gr.Dropdown(choices = list(prompts["prompt_name"]))

def generate_prompts(formatted_prompt, temperature, top_p, lines, voice, project, prompt_name):      
    p = formatted_prompt.format(lines=lines)
    if type(voice) is not type("") or type(project) is not type(""):
        gr.Warning("Please choose a project and voice")
        return "", None
    voice_uuid = re.search(r'\[(.*?)\]', voice).group(1)
    project_uuid = re.search(r'\[(.*?)\]', project).group(1)

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
    gr.Info("Claude finished generating line")
    
    resemble.generate(title=prompt_name, text=gen_line, project_uuid=project_uuid, voice_uuid=voice_uuid)
    gr.Info("Resemble finished generating audio")
    
    
    return gen_line, gr.Audio(value=f"./output/{prompt_name}.wav", interactive=True)

def reset():
    #load_saved()
    return "", gr.Audio(interactive=False)

def list_voices():
    global prompts
    voices = resemble.list_voices()
    projects = resemble.list_projects()
    r1 = [f"{v['name']} [{v['uuid']}]" for v in voices]
    r2 = [f"{v['name']}, [{v['uuid']}]" for v in projects]
    return gr.Dropdown(choices=r1), gr.Dropdown(choices=r2), list(prompts["prompt_name"])

with gr.Blocks(title="Adding Machine Prompt Editor", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("Edit prompt"):
            with gr.Row():
                drop = gr.Dropdown(choices = list(prompts["prompt_name"]), label="Choose stored prompt")
                prompt_name = gr.Textbox(label="Prompt name")                
            with gr.Row():
                main_prompt = gr.TextArea(label="Prompt template", value="", interactive=True)

            with gr.Row():
                with gr.Column():
                    scene_desc = gr.TextArea(label="Scene Description", value="", interactive=True)
                with gr.Column():
                    char_desc = gr.TextArea(label="Character Description", value="", interactive=True)
            with gr.Row():
                lines = gr.TextArea(label="Lines", value="", interactive=True)

            with gr.Row():
                    overwrite_btn = gr.Button(value="Overwrite selected prompt", variant='primary')  
                    save_btn = gr.Button(value="Save as new prompt", variant='primary')  
        with gr.Tab("Generate Lines") as gen_lines_tab:
            with gr.Row():
                drop2 = gr.Dropdown(choices = list(prompts["prompt_name"]), label="Choose stored prompt")
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.TextArea(label="Prompt template", interactive=False)
                    temperature = gr.Slider(label="Temperature (Higher = more creative, lower = more predictable response)", value=0.7, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    top_p = gr.Slider(label="Top P (Higher = larger set of words to choose from, lower = choose from the most likely words)", value=0.5, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                                       
                with gr.Column():
                    gen_lines = gr.TextArea(label="Lines", interactive=True)
                    project = gr.Dropdown(choices=["a", "b"], label="Resemble Project", interactive=True)
                    voice = gr.Dropdown(choices=["a", "b"], label="Voice", interactive=True)
            with gr.Row():
                generate_btn = gr.Button(value="Generate Line", variant="primary")
                reset_btn = gr.Button(value="Reset", variant="primary")
            with gr.Row():
                output = gr.TextArea(label="Claude Output", interactive=False, visible=True)  
                output_aud = gr.Audio(label="Resemble Output", type="filepath", interactive=False)
    drop.change(fn=select_prompt, inputs=drop, outputs=[prompt_name, main_prompt, scene_desc, char_desc, lines])
    drop2.change(fn=select_prompt2, inputs=drop2, outputs=[gen_prompt, gen_lines])
    overwrite_btn.click(fn=overwrite_prompt, inputs=[prompt_name, main_prompt, scene_desc, char_desc, lines], outputs=[drop])
    save_btn.click(fn=save_new, inputs=[prompt_name, main_prompt, scene_desc, char_desc, lines], outputs=[drop])
    generate_btn.click(fn=generate_prompts, inputs=[gen_prompt, temperature, top_p, gen_lines, voice, project, drop2], outputs=[output, output_aud])
    reset_btn.click(fn=reset, outputs=[output, output_aud])
    gen_lines_tab.select(fn=list_voices, inputs=[], outputs=[voice, project, drop2])
    
                    
demo.launch()