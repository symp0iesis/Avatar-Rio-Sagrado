import os
import librosa
import time
from STTEngine.STTEngineWhisper import STTEngine

# load model and processor for speech-to-text
stt_engine = STTEngine()

speech_data, sampling_rate = librosa.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.wav"), sr=16000)

# test speech-to-text engine for the inference time and the quality of the transcription
start_time = time.time()
text = stt_engine.speech_to_text(speech_data, sampling_rate)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(text)
