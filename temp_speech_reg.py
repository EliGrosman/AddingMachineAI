from live_transcription import listen2 as transcribe
with gr.Column():
                    transcript_lines = gr.TextArea(label="Transcribed Lines", interactive=False)
                    block_size = gr.Slider(label="Number of seconds per line (cannot be greater than total time it will listen to)", value=1, minimum=1, maximum=60, step=1, interactive=True)

                    num_secs = gr.Slider(label="Number of seconds to transcribe", value=5, minimum=1, maximum=60, step=0.1, interactive=True)
                    transcribe_btn = gr.Button(value="Transcribe", variant="primary")
                    #gen_info = gr.TextArea(label="Info", interactive=False)
                    
transcribe_btn.click(fn=transcribe, inputs=[num_secs, block_size], outputs=[transcript_lines])