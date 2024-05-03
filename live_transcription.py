import sounddevice as sd
import queue
import numpy as np
import whisper
from datetime import datetime

WHISPER_MODEL = "small" # change to base or tiny for faster predictions. change to medium or large for more accurate but slower preds.
q = queue.Queue()
def callback(indata, frame_count, time_info, status):
    q.put_nowait((indata.copy(), status))

fs = 16000 

def listen(num_secs):
    indata = sd.rec(num_secs * fs, samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    model = whisper.load_model(WHISPER_MODEL, download_root="models")

    indata_transformed = indata.flatten().astype(np.float32) / 32768.0
    result = model.transcribe(indata_transformed, language="english")
    print(result["text"])
    return result["text"]

def listen2(num_secs, secs_per_chunk):
    all_recs = []
    text = []
    start_ts = datetime.now()
    model = whisper.load_model(WHISPER_MODEL, download_root="models")
    with sd.InputStream(callback=callback, dtype='int16', channels=1, samplerate=16000, blocksize=16000*secs_per_chunk):
        while True:
            indata, _ = q.get()
            all_recs.append(indata)
            delta = datetime.now() - start_ts
            
            if delta.seconds >= num_secs:
                break

    for recording in all_recs:
        indata_transformed = recording.flatten().astype(np.float32) / 32768.0
        result = model.transcribe(indata_transformed, language="english")
        text.append(result["text"])
    return "\n".join(text)


