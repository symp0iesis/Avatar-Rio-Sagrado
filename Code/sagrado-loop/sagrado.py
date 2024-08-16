import time, os, threading, requests, sounddevice
import numpy as np
from numpy import array
import traceback

#To toggle modifications needed to make code work on Mayowa's computer
# on_mac = True


# if on_mac == True:
#     os.environ['KMP_DUPLICATE_LIB_OK']='True'


def init_stt():
    print('Initializing Speech-to-Text...')
    #Set up STT
    import torch
    import pyaudio
    from STT_Module.STTEngine.STTEngineFasterWhisper import STTEngine
    global stream, stt_engine, speech_data, CHUNK, SAMPLING_RATE, VAD_WINDOW_LENGTH

    speech_data = []
    SAMPLING_RATE = 16000
    VAD_WINDOW_LENGTH = 512 #1600
    # VAD_THRESHOLD = 0.4
    # MIN_SILENCE_DURATION_MS = 500
    # SPEECH_PAD_MS = 100
    CHUNK = 1024

    # load model and processor for speech-to-text
    stt_engine = STTEngine()

    # load model and utils for voice activity detection
    # model_vad, utils_vad = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
    # (_, _, _, VADIterator, _) = utils_vad
    # vad_iterator = VADIterator(model_vad, threshold=VAD_THRESHOLD, min_silence_duration_ms=MIN_SILENCE_DURATION_MS, speech_pad_ms=SPEECH_PAD_MS)

    # initialize listening device
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=VAD_WINDOW_LENGTH)

    print('Done.\n')

init_stt()


#Set up TTS
from soundfile import write as sf_write
import sounddevice as sd
from sounddevice import play, wait
import nltk.data
print(' Loading nltk tokenizer...')
os.environ['NLTK_DATA'] = 'TTS_Module/nltk_data/'
sentence_tokenizer = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')
# nltk.download('punkt')
print(' Done.')

def init_coqui_tts():
    print('Initializing Text-to-Speech...')
    from torch.cuda import is_available as cuda_is_available
    from TTS.api import TTS
    global split_text_into_sentences, text_to_speech

    def load_tts_model(device):
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    def split_text_into_sentences(text):
        sentences = sentence_tokenizer.tokenize(text)
        # sentences = nltk.sent_tokenize(text)
        for i in range(len(sentences)):
            if sentences[i][-1] == '.':
                sentences[i] = sentences[i][:-1]
        return sentences

    device = "cuda" if cuda_is_available() else "cpu"
    # device = "cpu"
    tts = load_tts_model(device)

    def text_to_speech(sentence):
        wav_data = tts.tts(text=sentence, language='pt', speaker_wav="TTS_Module/cr7.wav", split_sentences=False)
        return wav_data

def init_piper_tts(): 
    print('Initializing Text-to-Speech...')
    from piper import PiperVoice
    global split_text_into_sentences, text_to_speech
    HEADER_SIZE = 44
    SAMPLE_RATE = 21000
    SPEECH_SPEED = 0.835

    synthesize_args = {
        "speaker_id": None,
        "length_scale": None,
        "noise_scale": None,
        "noise_w": None,
        "sentence_silence": 0.5,
        }
    voice = PiperVoice.load('/home/avatar-rio-sagrado/piper_tts/pt_BR-faber-medium.onnx', '/home/avatar-rio-sagrado/piper_tts/pt_BR-faber-medium.onnx.json')

    def split_text_into_sentences(text):
        sentences = sentence_tokenizer.tokenize(text)
        # sentences = nltk.sent_tokenize(text)
        return sentences

    def text_to_speech(text):
        try:
            #Do this every time, or add generated audio to a pre-existing stream?
            stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
            stream.start()
            sentences = split_text_into_sentences(text)
            for sentence in sentences:
                for audio_bytes in voice.synthesize_stream_raw(sentence):
                    int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    stream.write(int_data)
            stream.stop()
            stream.close()

        except Exception as e:
            print("Error generating speech from text: ", Exception,  e)
            print(traceback.format_exc())


# Select TTS Backend to Use:
# init_coqui_tts()
init_piper_tts()


print('Done.\n')


# # To just test the TTS
# avatar_response = 'Olá, eu estou sempre fluindo, sempre renovando-me nas águas que percorrem o município de Morretes até a baía de Antonina. Como posso ajudar você hoje?'
# speak(avatar_response)


def init_retune():
    print('Setting up Retune communication utils...')
    #Utilities for Communicating with Retune Avatar

    global json_data, generate_response_url, headers

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

init_retune()

def avatar_response(speech_text):
    # print('Getting avatar response...')
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

    text_to_speech(avatar_response)

#Main loop. Listens for audio, transcribes it, passes it to Retune Avatar, 
#converts Avatar output text to speech.

# print("Listening for speech...")

avatar_mode = 'inactive'
listening=False

time.sleep(2.5)


def listen():
    global speech_data
    speech_data = np.array([])
    while listening == True:
        data_np = np.frombuffer(stream.read(CHUNK, dtype=np.float32))
        speech_data = np.append(speech_data, data_np)  
        # speech_data.append(data)
    print('Listening stopped.')


def respond(speech_data):
    print('Transcribing audio...')
    speech_text = stt_engine.speech_to_text(speech_data, SAMPLING_RATE)

    print('Transcription: ', speech_text)
    # if avatar_mode == 'active':
        # print('Transcription: ', speech_text)
    avatar_response(speech_text)
    
    # if (avatar_mode=='inactive') and ('água' in speech_text.lower() or 'agua' in speech_text.lower()):
    #     avatar_mode = 'active'
    #     print('Avatar activated')
        

while True:
    x = input('Press any key to listen...')
    listening = True
    listener_thread = threading.Thread(target=listen)
    listener_thread.start()
    x = input('Press any key to stop listening...')
    listening=False
    print('Speech data: ', len(speech_data), speech_data.shape)
    respond(speech_data)


# while True:
#     data_np = np.frombuffer(stream.read(VAD_WINDOW_LENGTH), dtype=np.float32)
#     speech_dict = vad_iterator(data_np, return_seconds=True)
#     if speech_dict:
#         # print('Speech dict: ', speech_dict)
#         if 'start' in speech_dict:
#             start_speech = True
#             end_speech = False
#         elif 'end' in speech_dict:
#             end_speech = True
#             start_speech = False
#     if start_speech and not end_speech:
#         speech_data = np.append(speech_data, data_np)        
#     elif end_speech and not start_speech:
#         speech_data = np.append(speech_data, data_np)
#         speech_text = stt_engine.speech_to_text(speech_data, SAMPLING_RATE)

#         speech_data = np.array([])
#         start_speech = False
#         end_speech = False

#         print('Transcription: ', speech_text)
#         if (avatar_mode=='inactive') and ('água' in speech_text.lower() or 'agua' in speech_text.lower()):
#             avatar_mode = 'active'
#             print('Avatar activated')
#             continue

#         # print('Getting avatar response...')
#         # speech_text = input('Enter input: ')

#         if avatar_mode == 'active':
#             # print('Transcription: ', speech_text)
#             avatar_response(speech_text)

