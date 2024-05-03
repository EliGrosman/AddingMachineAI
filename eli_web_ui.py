import gradio as gr
import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np
from resemble import Resemble
import whisper

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
Resemble.api_key("cE3HL6w4rVCF2l8WqBvAHAtt")

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


def generate_prompts(formatted_prompt, lines, temperature, top_p, generated_lines_storage):
    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": formatted_prompt.format(lines=lines, next_line="None")
            }
        ],
        temperature=temperature,
        top_p=top_p,
        model="claude-3-opus-20240229"
    )

    return message.content[0].text


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
    num_to_generate = sum([1 for i in range(len(split_lines)) if len(
        split_lines[i].split(":")[1].strip()) == 0])

    info = f"""
    Number of lines to generate: {num_to_generate}
    """

    return formatted_prompt, lines, info


def reset():
    # load_saved()
    return "", 0, ""

# creates clip
# returns resemble link


def create_clip(formatted_prompt, lines, temperature, top_p, body, audio):
    print(audio)
    print(f"TYPES: {type(audio)}, {type(audio[0])}, {type(audio[1])}")

    model = whisper.load_model("base")
    options = whisper.DecodingOptions(language="en")
    result = model.transcribe(audio[1].astype(np.float32), fp16=False)
    new_lines = lines + "\nAI:" + result["text"]  # TODO check this
    print(new_lines)
    text = generate_prompts(formatted_prompt, new_lines,
                            temperature, top_p, body)[2]
    print(text)

    response = Resemble.v2.clips.create_sync(
        project_uuid="7848927a",
        voice_uuid="34f51533",
        body=text,
        is_public=False,
        is_archived=False,
        title=None,
        sample_rate=None,
        output_format=None,
        precision=None,
        include_timestamps=None,
        raw=True
    )

    if response['success']:
        clip = response['item']
        clip_uuid = clip['uuid']
        clip_url = clip['audio_src']
        clip_raw = clip['raw_audio']
        print(clip_url)
        return gr.Audio(value=clip_url)
    else:
        print(response)
        print("Response was unsuccessful.")


input_audio = gr.Audio(type="numpy", sources=["microphone"], waveform_options=gr.WaveformOptions(
    waveform_color="#01C6FF",
    waveform_progress_color="#0066B4",
    skip_length=2,
    show_controls=False,
),
)

with gr.Blocks(title="Adding Machine Prompt Editor", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("Edit prompt"):
            with gr.Row():
                main_prompt = gr.TextArea(
                    label="Prompt template", value=templates["main_prompt"], interactive=True)

            with gr.Row():
                with gr.Column():
                    scene_desc = gr.TextArea(
                        label="Scene Description", value=templates["scene_desc"], interactive=True)
                with gr.Column():
                    char_desc = gr.TextArea(
                        label="Character Description", value=templates["char_desc"], interactive=True)
            with gr.Row():
                lines = gr.TextArea(
                    label="Lines", value=templates["lines"], interactive=True)

            with gr.Row():
                save_btn = gr.Button(value="Save prompt", variant='primary')
        with gr.Tab("Generate Lines") as gen_lines_tab:
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.TextArea(
                        label="Prompt template", interactive=False, visible=False)
                with gr.Column():
                    gen_lines = gr.TextArea(
                        label="Lines", interactive=False, visible=False)
                    gen_info = gr.TextArea(
                        label="Info", interactive=False, visible=False)

                    temperature = gr.Slider(label="Temperature (Higher = more creative, lower = more predictable response)",
                                            value=0.7, minimum=0.0, maximum=1.0, step=0.01, interactive=True, visible=False)
                    top_p = gr.Slider(label="Top P (Higher = larger set of words to choose from, lower = choose from the most likely words)",
                                      value=0.5, minimum=0.0, maximum=1.0, step=0.01, interactive=True, visible=False)

            with gr.Row():
                generate_btn = gr.Button(
                    value="Generate Next Line", variant="primary")
                reset_btn = gr.Button(value="Reset", variant="primary")
            with gr.Row():
                output = gr.TextArea(
                    label="Generated Prompts", interactive=False, visible=False)
                # num_generated = gr.Number(value=0, visible=False)
                generated_lines_storage = gr.TextArea(
                    interactive=False, value=None, visible=False)
            with gr.Row():
                play_btn = gr.Interface(fn=create_clip, submit_btn="Submit", inputs=[
                    gen_prompt, gen_lines, temperature, top_p, generated_lines_storage, input_audio], outputs=["audio"])

    save_btn.click(fn=save_prompts, inputs=[
                   main_prompt, scene_desc, char_desc, lines])
    gen_lines_tab.select(fn=load_saved, inputs=[], outputs=[
                         gen_prompt, gen_lines, gen_info])
    generate_btn.click(fn=generate_prompts, inputs=[
                       gen_prompt, gen_lines, temperature, top_p, generated_lines_storage], outputs=[output, generated_lines_storage])
    reset_btn.click(fn=reset, outputs=[output, generated_lines_storage])

demo.launch()
