from time import time, sleep

#Set up STT

import torch
import pyaudio
import numpy as np

#Modification to make code work on Mayowa's computer.
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


print('Initializing Speech-to-Text...')

from STT_Module.STTEngine.STTEngineFasterWhisper import STTEngine

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

print('Done.\n')




#Set up TTS

# import os
# from soundfile import write as sf_write
# from torch.cuda import is_available as cuda_is_available
# import threading
# from numpy import array
# from TTS.api import TTS
# from sounddevice import play, wait
# import nltk
# nltk.download('punkt')


# print('Initializing Text-to-Speech...')

# @st.cache_resource
# def load_tts_model(device):
#     return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


# def split_text_into_sentences(text):
#     sentences = nltk.sent_tokenize(text)
#     for i in range(len(sentences)):
#         if sentences[i][-1] == '.':
#             sentences[i] = sentences[i][:-1]
#     return sentences


# def play_audio():
#     global audio_queue
#     while True:
#         sleep(0.3)
#         if len(audio_queue) > 0:
#             wav_data = audio_queue.pop(0)
#             play(array(wav_data), 24000)
#             wait()


# device = "cuda" if cuda_is_available() else "cpu"
# # device = "cpu"
# tts = load_tts_model(device)

# # Global audio queue to store the audio data
# audio_queue = []
# file_data = []

# # Start the audio playback thread
# audio_thread = threading.Thread(target=play_audio)
# audio_thread.daemon = True
# audio_thread.start()


# def speak(input_text):
#     sentences = split_text_into_sentences(input_text)
#     for sentence in sentences:
#         wav_data = tts.tts(text=sentence, language=language_code, speaker_wav="cr7.wav", split_sentences=False)
#         audio_queue.append(wav_data)
#         file_data.extend(wav_data)

#     sf_write("output.wav", array(file_data), 24000)
#     # st.audio("output.wav", format="audio/wav")
#     # os.remove("output.wav")


# print('Done.\n')



print('Setting up Retune communication utils...')
#Utilities for Communicating with Retune Avatar
import requests

create_thread_url = 'https://retune.so/api/chat/11ee2b4b-d054-47f0-9771-310dd2eca1c4/new-thread'
generate_response_url = 'https://retune.so/api/chat/11ee2b4b-d054-47f0-9771-310dd2eca1c4/response'


headers = {
    'Content-Type': 'application/json',
    'X-Workspace-API-Key': '11ee2f36-c329-7e80-9c79-ff4ddae14db0',
}


#Create new conversation thread
print('\n Creating new conversation thread with Retune avatar...')
response = requests.get(
    create_thread_url,
    headers=headers,
)
print(' Done.\n')

#Could implement simultaneous conversations with multiple threads at some point,
# depending on how this is planned to be deployed.
# print('Response: ', response.json())
threadId = response.json()['threadId']

json_data = {
    'threadId': threadId, #'11ef1c61-8795-5150-af65-e1c12ce1ce56',
    'input': 'Hello! Please speak English to me. What are you about?',
}

print('Done.\n')


#Main loop. Listens for audio, transcribes it, passes it to Retune Avatar, 
#converts Avatar output text to speech.

print("Listening for speech...")

while True:
    # The exception_on_overflow = False halps the code work on Mayowa's computer.
    # data_np = np.frombuffer(stream.read(VAD_WINDOW_LENGTH, exception_on_overflow = False), dtype=np.float32)
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

    speech_data = np.append(speech_data, data_np)        
    if end_speech and not start_speech:
        speech_data = np.append(speech_data, data_np)
        # start_time = time()
        speech_text = stt_engine.speech_to_text(speech_data, SAMPLING_RATE)
        # end_time = time()
        print(f"Transcription: {speech_text}")
        # print(f"Time taken: {end_time - start_time:.2f} seconds\n")

        print('Getting avatar response...')

        json_data['input'] = speech_text
        response = requests.post(
            generate_response_url,
            headers=headers,
            json=json_data,
        )

        #Issue communicating with Retune, and obtaining avatar response
        if response.status_code != 200:
            avatar_response = "Retune Error." #In Portuguese? What would Danilo want?
        else:
            avatar_response = response.json()['response']['value']

        print('Avatar response: ', avatar_response)
        # speak(avatar_response)
        # speech_data = np.array([])
        # start_speech = False
        # end_speech = False






