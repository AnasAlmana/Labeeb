from openai import OpenAI 
import openai
import os
import base64
import pyaudio
import wave
import tempfile
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path
import datetime
import whisper  
import audioop
from termcolor import colored

MODEL = "gpt-4o"
openai.api_key = #OPENAI_KEY

client = OpenAI(api_key="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")

VIDEO_PATH = "test.mp4"

def gpt4o_chat(query):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "اسمك هو لبيب، انت مساعد ذكاء إصطناعي تتحدث العربية بطلاقة، تم تطويرك بواسطة غسّان الوَرد، أنَس ال مانع و محمد ال سليم بإشراف من الاستاذ عبدالله الوابل والاستاذ بدر الصبيحي وهم مجموعة في سدايا المستقبل"},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=200,
        top_p=1,
        presence_penalty=0,
        frequency_penalty=0
    )
    return response.choices[0].message.content.strip()

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, user_input):
    base64_image = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"هذا سؤال المستخدم المتعلق بالصورة المرفقة: {user_input}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ], 
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def text_to_speech_and_play(text):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpfile:
        speech_file_path = tmpfile.name

    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format='mp3'
    )

    with open(speech_file_path, 'wb') as f:
        f.write(response.content)

    audio = AudioSegment.from_mp3(speech_file_path)
    play(audio)
    Path(speech_file_path).unlink()

whisper_model = whisper.load_model("medium")

def transcribe_with_whisper(audio_file):
    result = whisper_model.transcribe(audio_file)
    transcription = result["text"]
    return transcription.strip()

def record_audio(file_path, silence_threshold=1000, speech_threshold=1000, chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    frames = []
    silence_count = 0
    silence_frames = int(rate / chunk_size * 1.5)  
    speech_frames = int(rate / chunk_size * 0.3)  

    print(colored("Waiting for speech...", "yellow"))
    
    while True:
        data = stream.read(chunk_size)
        rms = audioop.rms(data, 2)

        if rms > speech_threshold:
            print(colored("recording...", "green"))
            break

    frames.append(data)

    while True:
        data = stream.read(chunk_size)
        frames.append(data)

        rms = audioop.rms(data, 2)
        if rms < silence_threshold:
            silence_count += 1
            if silence_count > silence_frames:
                break
        else:
            silence_count = 0
            
    print(colored("Recording stopped.", "red"))
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    folder_path = "./images"
    os.makedirs(folder_path, exist_ok=True)

    print("Labeeb Observing..")
    while True:
        #audio_file = "temp.wav"
        #record_audio(audio_file)
        #user_input = transcribe_with_whisper(audio_file)
        #print(user_input)
        #os.remove(audio_file)
        user_input = "اعطني نبذة عنك"

        if "exit" in user_input.lower():
            break

        feedback = gpt4o_chat(f"سؤال المستخدم: {user_input}\n\n، اجب عليه بمعرفتك")
        text_to_speech_and_play(feedback)

if __name__ == "__main__":
    main()


## SUDOKU
"""
def audioshit(req, res):
    ## Step 1: Get audio from user
    wshfeekakhook = req.audiofile

    ## Step 2: GPT will response to the audio file with another audio file
    responseToAudio = gpt_get_audiofile(wshfeekakhook)

    ## Step 3: Send it back to the user
    res.send(responseToAudio)"""

