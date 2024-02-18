from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re


class STTEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # load model and processor for speech-to-text using OpenAI's Whisper model, tiny version
        self.processor_stt = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model_stt = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.model_stt.config.forced_decoder_ids = None

    def speech_to_text(self, speech_data, sampling_rate):
        input_features = self.processor_stt(speech_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model_stt.generate(input_features)
        transcription = self.processor_stt.batch_decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pattern = r"\s*<\|\d+\.\d+\|>\s*"
        text = re.sub(pattern, "", transcription[0])
        return text
