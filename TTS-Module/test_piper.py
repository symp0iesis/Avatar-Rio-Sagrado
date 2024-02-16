import os
import streamlit as st
from soundfile import write as sf_write
import threading
import numpy as np
import requests
from time import sleep
from sounddevice import play, wait
import nltk
nltk.download('punkt')

HEADER_SIZE = 44
SAMPLE_RATE = 21000
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


st.set_page_config(page_title="Text to Speech Raspberry Pi", page_icon="🔊", layout="centered", initial_sidebar_state="collapsed")
st.title('Text to Speech Raspberry Pi')
input_text = st.text_input('Enter the text you want to convert to speech in Portuguese')
submit_button = st.button('Convert to speech')

# Global audio queue to store the audio data
audio_queue = []
file_data = []

# Start the audio playback thread
audio_thread = threading.Thread(target=play_audio)
audio_thread.daemon = True
audio_thread.start()

if input_text and submit_button:
    sentences = split_text_into_sentences(input_text)
    for sentence in sentences:
        print(sentence)
        wav_data = text_to_speech(URL, sentence)
        if wav_data is not None:
            audio_queue.append(wav_data)
            file_data.extend(wav_data)
        else:
            st.error("Failed to convert text to speech")
            raise Exception("Failed to convert text to speech")

    sf_write("output.wav", file_data, SAMPLE_RATE)
    st.audio("output.wav", format="audio/wav")
    # os.remove("output.wav")
