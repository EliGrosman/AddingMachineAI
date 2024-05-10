import gradio as gr
import pandas as pd
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from resemble_wrapper import Resemble_Wrapper
import re

load_dotenv()

client = Anthropic(api_key = os.environ["ANTHROPIC_API_KEY"])
resemble = Resemble_Wrapper(os.environ["RESEMBLE_API_KEY"])

if os.path.isfile("./saved_prompts.csv"):
    prompts = pd.read_csv("./saved_prompts.csv")
else:
    prompts = pd.DataFrame({"prompt_name": [], "main_prompt": []})
    prompts.to_csv("./saved_prompts.csv", index = False)

def select_prompt(choice):
    global prompts
    if type(choice) is not type([]):
        row = prompts[prompts["prompt_name"] == choice].iloc[0]
        return row["prompt_name"], row["main_prompt"]
    return "", ""

def overwrite_prompt(prompt_name, main_prompt):
    global prompts
    prompts.loc[prompts["prompt_name"] == prompt_name, "main_prompt"] = main_prompt
    
    prompts.to_csv("./saved_prompts.csv", index = False)
    
    gr.Info(f"Overwrote prompt {prompt_name}")
    return gr.Dropdown(choices = list(prompts["prompt_name"]))
    
def save_new(prompt_name, main_prompt):
    global prompts
    if prompt_name in list(prompts["prompt_name"]):
        gr.Warning(f"Prompt '{main_prompt}' already exists")
    else:
        new_r = pd.DataFrame({"prompt_name": [prompt_name], "main_prompt": [main_prompt]})
        prompts = pd.concat([prompts, new_r])
        prompts.to_csv("./saved_prompts.csv", index = False)
        
        gr.Info(f"Saved prompt {prompt_name}")
    return gr.Dropdown(choices = list(prompts["prompt_name"]))

def process_history(usr_prompt, history, voice, project, sys_prompt):
    error = False
    if type(voice) is not type("") or type(project) is not type(""):
        gr.Warning("Please choose a project and voice")
        error = True
    if len(sys_prompt) == 0:
        gr.Warning("Please select a system prompt")
        error = True
        
    if error:
        raise Exception("An error has occured")
        
    history.append((usr_prompt, None))
    return usr_prompt, "", history

def history_to_messages(history):
    out = []
    for msg in history:
        print(msg[1])
        if len(msg[1]) > 0:
            out.append({"role": "user", "content": msg[0]})
            out.append({"role": "assistant", "content": msg[1]})
        else:
            out.append({"role": "user", "content": msg[0]})
        
    return out

def send_prompt(sys_prompt, usr_prompt, history, temperature, top_p):
    print(history)
    
    
    history[-1][1] = ""
    gr.Info("Claude is thinking...")
    
    messages = history_to_messages(history)
    print(messages)
    message = client.messages.create(
        max_tokens=1024,
        system=sys_prompt,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        model="claude-3-opus-20240229",
        stream=True
    )
    out = ""
    for c in message:
        if c.type == 'content_block_delta':
            out += c.delta.text
            history[-1][1] += c.delta.text
            yield history
    
def speak(chat, prompt_name, voice, project):
    voice_uuid = re.search(r'\[(.*?)\]', voice).group(1)
    project_uuid = re.search(r'\[(.*?)\]', project).group(1)
    gr.Info("Resemble is processing...")
    try:
        resemble.generate(title=prompt_name, text=chat[-1][1], project_uuid=project_uuid, voice_uuid=voice_uuid)
    except:
        raise Exception("An error has occured, something went wrong on Resemble's end. Please try again")
    
    return gr.Audio(value=f"./output/{prompt_name}.wav", interactive=True)
    

def reset():
    #load_saved()
    return "", gr.Audio(interactive=False)

def load_gen_tab():
    global prompts
    gr.Info("Loading Resemble projects and voices. This takes a sec")
    voices = resemble.list_voices()
    projects = resemble.list_projects()
    r1 = [f"{v['name']} [{v['uuid']}]" for v in voices]
    r2 = [f"{v['name']}, [{v['uuid']}]" for v in projects]
    return gr.Dropdown(choices=r1), gr.Dropdown(choices=r2), gr.Dropdown(choices=list(prompts["prompt_name"]))

with gr.Blocks(title="Adding Machine Prompt Editor", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("Edit prompt"):
            with gr.Row():
                drop = gr.Dropdown(choices = list(prompts["prompt_name"]), label="Choose stored prompt")
                prompt_name = gr.Textbox(label="Prompt name")                
            with gr.Row():
                main_prompt = gr.TextArea(label="Prompt template", value="", interactive=True)

            with gr.Row():
                    overwrite_btn = gr.Button(value="Overwrite selected prompt", variant='primary')  
                    save_btn = gr.Button(value="Save as new prompt", variant='primary')  
        with gr.Tab("Generate Lines") as gen_lines_tab:
            tmp_usr_prompt = gr.Textbox(interactive=False, visible=False)
            with gr.Row():
                drop2 = gr.Dropdown(choices = list(prompts["prompt_name"]), label="Choose stored prompt")
                selected_label = gr.Label(label="Selected Prompt", visible = False)
                prompt_storage = gr.Textbox(label="System prompt", visible=True, interactive=False)
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        chat = gr.Chatbot()
                    with gr.Row():
                        user_prompt = gr.Textbox(label="Enter prompt")
                    with gr.Row():
                        output_aud = gr.Audio(label="Resemble Output", type="filepath", interactive=False)
                    with gr.Row():
                        send_btn = gr.Button(value="Send", variant="primary")
                        clear_btn = gr.Button(value="Clear", variant="primary")
                with gr.Column(scale=1):
                    with gr.Row(variant="panel"):
                        project = gr.Dropdown(choices=["a", "b"], label="Resemble Project", interactive=True)
                        voice = gr.Dropdown(choices=["a", "b"], label="Voice", interactive=True)
                    with gr.Row():
                        temperature = gr.Slider(label="Temperature (Higher = more creative, lower = more predictable response)", value=0.7, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    with gr.Row():
                        top_p = gr.Slider(label="Top P (Higher = larger set of words to choose from, lower = choose from the most likely words)", value=0.5, minimum=0.0, maximum=1.0, step=0.01, interactive=True)
    drop.change(fn=select_prompt, inputs=drop, outputs=[prompt_name, main_prompt])
    drop2.change(fn=select_prompt, inputs=drop2, outputs=[selected_label, prompt_storage])
    overwrite_btn.click(fn=overwrite_prompt, inputs=[prompt_name, main_prompt], outputs=[drop])
    save_btn.click(fn=save_new, inputs=[prompt_name, main_prompt], outputs=[drop])
    gen_lines_tab.select(fn=load_gen_tab, inputs=[], outputs=[voice, project, drop2])
    
    user_prompt.submit(fn=process_history, inputs=[user_prompt, chat, voice, project, prompt_storage], outputs=[tmp_usr_prompt, user_prompt, chat]).success(
        fn=send_prompt, inputs=[prompt_storage, tmp_usr_prompt, chat, temperature, top_p], outputs=[chat]
    ).success(fn=speak, inputs=[chat, drop2, voice, project], outputs=[output_aud])
    send_btn.click(fn=process_history, inputs=[user_prompt, chat, voice, project, prompt_storage], outputs=[tmp_usr_prompt, user_prompt, chat]).success(
        fn=send_prompt, inputs=[prompt_storage, tmp_usr_prompt, chat, temperature, top_p], outputs=[chat]
    ).success(fn=speak, inputs=[chat, drop2, voice, project], outputs=[output_aud])
    clear_btn.click(lambda: (None, None, None), None, [chat, user_prompt, output_aud])
    
                    
demo.launch()