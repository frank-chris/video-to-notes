from moviepy.editor import VideoFileClip

class AudioExtractor:
    def __init__(self, video_file_path, audio_file_path):
        self.video_file_path = video_file_path
        self.audio_file_path = audio_file_path

    def extract_audio(self):
        video = VideoFileClip(self.video_file_path)
        audio = video.audio
        audio.write_audiofile(self.audio_file_path)
        video.close()


if __name__ == "__main__":
    video_path = 'C:\\Users\\shiva\\Desktop\\Spring 2024\\video-to-notes\\src\\cse290-w24-lec8.mp4'
    audio_path = 'cse290-w24-lec8-audio.mp3'
    extractor = AudioExtractor(video_path, audio_path)
    extractor.extract_audio()
