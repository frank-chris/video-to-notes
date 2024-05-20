import whisper

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_file_path):
        result = self.model.transcribe(audio_file_path)
        return result

    def save_transcription(self, transcription_result, output_file_path):
        with open(output_file_path, "w") as f:
            f.write("Full Transcription:\n")
            f.write(transcription_result["text"])
            f.write("\n\nDetailed Segments with Timestamps:\n")
            for segment in transcription_result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                f.write(f"{start_time:.2f} - {end_time:.2f}: {text}\n")
    

if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    audio_path = "cse290-w24-lec8-audio.mp3"
    transcription_result = transcriber.transcribe(audio_path)
    output_file_path = "transcription_result_with_timestamp.txt"
    transcriber.save_transcription(transcription_result, output_file_path)
    print("Transcription saved to", output_file_path)

