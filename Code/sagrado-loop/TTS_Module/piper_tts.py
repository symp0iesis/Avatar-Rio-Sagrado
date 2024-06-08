#Mayowa: File I set up to test Piper TTS without having to go through Streamlit.

print('Initializing...')
import os
from soundfile import write as sf_write
import threading
import numpy as np
import requests
from time import sleep
from sounddevice import play, wait
import pyrubberband
import nltk
# print(' Downloading nltk punkt...')
# nltk.download('punkt')
# print(' Done.')

HEADER_SIZE = 44
SAMPLE_RATE = 21000
SPEECH_SPEED = 0.835
URL = "http://localhost:5000"


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def bytes_to_sound(data):
    sound_data = np.frombuffer(data, dtype=np.int16, count=len(data) // 2).astype(np.float64) / 32768.0
    return sound_data


def text_to_speech(url, text):
    resp = requests.get(url, params={'text': text})
    if resp.status_code == 200:
        return bytes_to_sound(resp.content[HEADER_SIZE:])
    return None


def play_audio():
    global audio_queue
    while True:
        sleep(0.5)
        if len(audio_queue) > 0:
            wav_data = audio_queue.pop(0)
            play(np.array(wav_data), SAMPLE_RATE)
            wait()

# Global audio queue to store the audio data
audio_queue = []
file_data = []

# Start the audio playback thread
audio_thread = threading.Thread(target=play_audio)
audio_thread.daemon = True
audio_thread.start()

def process(input_text):
    print('Received text input: ', input_text)
    sentences = split_text_into_sentences(input_text)
    for sentence in sentences:
        print('Processing: ', sentence)
        print('Obtaining audio data from Piper Server...')
        wav_data = text_to_speech(URL, sentence)
        print('Done.')
        if wav_data is not None:
            slowed_wav_data = pyrubberband.time_stretch(wav_data, SAMPLE_RATE, SPEECH_SPEED)
            audio_queue.append(slowed_wav_data)
            file_data.extend(slowed_wav_data)
            print('Added audio data to queue.')
        else:
            raise Exception("Failed to convert text to speech")

    sf_write("output.wav", file_data, SAMPLE_RATE)
    # os.remove("output.wav")

print('Done.')

process('Ola, como estas?')
