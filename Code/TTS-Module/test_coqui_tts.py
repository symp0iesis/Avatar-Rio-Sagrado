import os
import streamlit as st
from soundfile import write as sf_write
from torch.cuda import is_available as cuda_is_available
import threading
from numpy import array
from time import sleep
from TTS.api import TTS
from sounddevice import play, wait
import nltk
nltk.download('punkt')


@st.cache_resource
def load_tts_model(device):
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    for i in range(len(sentences)):
        if sentences[i][-1] == '.':
            sentences[i] = sentences[i][:-1]
    return sentences


def play_audio():
    global audio_queue
    while True:
        sleep(0.3)
        if len(audio_queue) > 0:
            wav_data = audio_queue.pop(0)
            play(array(wav_data), 24000)
            wait()


st.set_page_config(page_title="Text to Speech", page_icon="🔊", layout="centered", initial_sidebar_state="collapsed")
st.title('Text to Speech')

device = "cuda" if cuda_is_available() else "cpu"
# device = "cpu"
tts = load_tts_model(device)

language_dict = {'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'tr': 'Turkish', 'ru': 'Russian', 'nl': 'Dutch', 'cs': 'Czech', 'ar': 'Arabic', 'zh-cn': 'Chinese', 'hu': 'Hungarian', 'ko': 'Korean', 'ja': 'Japanese', 'hi': 'Hindi'}
language_selected = st.selectbox('Select language', list(language_dict.values()))
language_code = list(language_dict.keys())[list(language_dict.values()).index(language_selected)]
input_text = st.text_input('Enter the text you want to convert to speech')
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
        wav_data = tts.tts(text=sentence, language=language_code, speaker_wav="cr7.wav", split_sentences=False)
        audio_queue.append(wav_data)
        file_data.extend(wav_data)

    sf_write("output.wav", array(file_data), 24000)
    st.audio("output.wav", format="audio/wav")
    # os.remove("output.wav")
