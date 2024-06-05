from openai import OpenAI

OpenAI.api_key = 'sk-proj-vqWVe0mhRxn8m9KgdMLLT3BlbkFJnuQcwdyh68jYUX5DYwuw'
client = OpenAI(api_key="sk-proj-vqWVe0mhRxn8m9KgdMLLT3BlbkFJnuQcwdyh68jYUX5DYwuw")


def voice_gpt4o():

    audio_file= open('audio/input.mp3', "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    print(transcription.text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                
            {"role": "system", "content": " اسمك هو لبيب، انت مساعد ذكاء إصطناعي تتحدث العربية بطلاقة، تم تطويرك بواسطة غسّان الوَرد، أنَس ال مانع و محمد ال سليم بإشراف من الاستاذ عبدالله الوابل والاستاذ بدر الصبيحي وانتم مجموعة في سدايا المستقبل"},
            {"role": "user", "content": f" {transcription.text}"},
        ]
    )

    print(response.choices[0].message.content)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=response.choices[0].message.content,
        response_format='mp3'
    )

    with open("responses/output.mp3", 'wb') as f:
            f.write(response.content)


if __name__ == '__main__':
      voice_gpt4o()