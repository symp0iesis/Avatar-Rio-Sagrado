import torch
from time import time
import pyaudio
import numpy as np
from STTEngine.STTEngineFasterWhisper import STTEngine

SAMPLING_RATE = 16000
VAD_WINDOW_LENGTH = 1600
VAD_THRESHOLD = 0.4
MIN_SILENCE_DURATION_MS = 500
SPEECH_PAD_MS = 100

# load model and processor for speech-to-text
stt_engine = STTEngine()

# load model and utils for voice activity detection
model_vad, utils_vad = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
(_, _, _, VADIterator, _) = utils_vad
vad_iterator = VADIterator(model_vad, threshold=VAD_THRESHOLD, min_silence_duration_ms=MIN_SILENCE_DURATION_MS, speech_pad_ms=SPEECH_PAD_MS)

# initialize listening device
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=VAD_WINDOW_LENGTH)
speech_data = np.array([])
start_speech = False
end_speech = False
print("Listening for speech...")

while True:
    data_np = np.frombuffer(stream.read(VAD_WINDOW_LENGTH), dtype=np.float32)
    speech_dict = vad_iterator(data_np, return_seconds=True)
    if speech_dict:
        print(speech_dict)
        if 'start' in speech_dict:
            start_speech = True
            end_speech = False
        elif 'end' in speech_dict:
            end_speech = True
            start_speech = False
    if start_speech and not end_speech:
        speech_data = np.append(speech_data, data_np)
    elif end_speech and not start_speech:
        speech_data = np.append(speech_data, data_np)
        start_time = time()
        speech_text = stt_engine.speech_to_text(speech_data, SAMPLING_RATE)
        end_time = time()
        print(f"Transcription: {speech_text}")
        print(f"Time taken: {end_time - start_time:.2f} seconds\n")
        speech_data = np.array([])
        start_speech = False
        end_speech = False
