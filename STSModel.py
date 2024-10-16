# 1 Speech to text
import librosa
# import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
# from deep_translator import GoogleTranslator
from googletrans import Translator
# import pyttsx3
import os
import soundfile as sf
from pocketsphinx import Decoder
from gtts import gTTS
base_dir = os.path.dirname(__file__)  # Get the current script directory
cache_dir = os.path.join(base_dir, 'pocketsphinx_cache', 'en-us')
model_dir = cache_dir
dictionary_path = os.path.join(cache_dir, 'cmudict-en-us.dict')


# Set a cache directory where the model will be stored after download
# cache_dir = "./my_model_cache"

# Load the processor and model using the cache directory
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)

def preprocess_audio(file_path, target_sr=16000):
    """Load and normalize the audio"""
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = librosa.util.normalize(audio)
    return audio, sr

def speech_to_text(file_path):
    """Convert speech to text"""
    # Load and preprocess the audio
    audio, sr = preprocess_audio(file_path)
    
    temp_audio_path = 'temp_audio.wav'
    sf.write(temp_audio_path, audio, sr)

    config = Decoder.default_config()
    config.set_string('-hmm', model_dir)  # Path to acoustic model
    config.set_string('-dict', dictionary_path)  # Path to dictionary
    # print(temp_audio_path)
    # Start decoding from an audio file
    decoder = Decoder(config)
    decoder.start_utt()
# Read audio file in chunks
    with open(temp_audio_path, 'rb') as audio_file:
        while True:
            buf = audio_file.read(1024)  # Read audio in chunks
            if not buf:
                break
            decoder.process_raw(buf, False, False)
    decoder.end_utt()

# Get recognized text
    transcription = decoder.hyp().hypstr
    os.remove(temp_audio_path)
    return transcription

    # Prepare the audio input for the model using the processor
    # input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values
   
    # Perform the transcription
    # with torch.no_grad():
        # logits = model(input_values).logits
   
    # Get the predicted token IDs and decode them to text
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.decode(predicted_ids[0])
    # return transcription

# def clean_transcription(text):
    """Clean the transcription by removing unwanted characters"""
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # text = text.lower()
    # return text

# 2. Text to Text
from deep_translator import GoogleTranslator

def translate_text(text: str, target_lang: str) -> str:
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return f"{translated_text}"
    except Exception as e:
        return f"Error during translation: {str(e)}"
    
# 3. Text to speech 
def text_to_speech(text: str, language: str,output_filename: str = 'output_audio.wav'):
    
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_filename)
    return output_filename
    # engine = pyttsx3.init()
    
    # Set the properties for the voice based on language
    # voices = engine.getProperty('voices')
    
    # Change the voice based on the selected language


    # translator = Translator()

    # Translate the text to the target language (French)
    # translated_text = translator.translate(text, dest=language).text
    # print(f"Translated text: {translated_text}")

    # Initialize TTS engine
    # engine = pyttsx3.init()

    # Define voice mapping
    # voices = {
    #     'en': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Zira Desktop',  # English
    #     'en_US': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Zira Desktop',  # American English
    #     'en_GB': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Hazel Desktop',  # British English
    #     'fr': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Henri Desktop',  # French
    #     'es': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Helena Desktop',  # Spanish
    #     'de': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Mark Desktop',  # German
    #     'it': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\SAPI\\Voices\\Tokens\\Microsoft Carla Desktop',  # Italian
    # }

    # Set the voice based on the selected language
    # if language in voices:
    #     engine.setProperty('voice', voices[language])
    # else:
    #     print(f"No voice found for language: {language}. Using default voice.")
    
    # rate = engine.getProperty('rate')
    # engine.setProperty('rate', rate - 50)

    # Save speech to a file
    # engine.save_to_file(translated_text, output_filename)
    # engine.runAndWait()
    
    # return output_filename
