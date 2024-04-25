import whisper

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file_path):
        result = self.model.transcribe(audio_file_path)
        return result


if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    audio_path = "engraft.mp3"
    transcription_result = transcriber.transcribe(audio_path)
    print(transcription_result["text"])
    print(transcription_result)
