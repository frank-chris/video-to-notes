from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.audio_to_text import WhisperTranscriber
from src.question_answering import TranscriptProcessor
from src.video_to_audio import AudioExtractor 
from src.text_to_notes import TextToNotes

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
input_file_path = None
notes = None
chat = []

def load_models():
    whisper_transcriber = WhisperTranscriber()
    transcript_processor = TranscriptProcessor(pinecone_api_key=os.getenv('PINECONE_API_KEY'), 
                                               pinecone_index_name='dl-ai')
    audio_extractor = AudioExtractor()
    ckpt_dir = "../llama3/Meta-Llama-3-8B-Instruct/"
    tokenizer_path = "../llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    llama = TextToNotes(ckpt_dir, tokenizer_path)

    return whisper_transcriber, transcript_processor, audio_extractor, llama

def format_notes(notes_dictionary):
    summary = notes_dictionary['summary'].replace('\n\n', '<br>')
    summary = summary.replace('<h3>', '<br><h3 class="text-lg font-medium">')
    if notes_dictionary['takeaways'] is not None:
        takeaways = '<br><br><h3 class="text-lg font-medium">Takeaways</h3>' + notes_dictionary['takeaways'].replace('\n', '')
        takeaways = takeaways.replace('<ul>', '<ul class="list-disc pl-5">')
    else:
        takeaways = ''
    if notes_dictionary['glossary'] is not None:
        glossary = '<br><h3 class="text-lg font-medium">Glossary</h3>' + notes_dictionary['glossary'].replace('\n', '')
        glossary = glossary.replace('<ul>', '<ul class="list-disc pl-5">')
    else:
        glossary = ''
    return summary + takeaways + glossary

@app.route('/', methods=['GET', 'POST'])
def home():
    global notes
    global chat
    global input_file_path
    global transcript_processor
    if request.method == 'POST':
        if 'lecturefile' in request.files:
            input_file = request.files['lecturefile']
            glossary = True if request.form.get('glossary') == 'on' else False
            takeaways = True if request.form.get('takeaways') == 'on' else False
            keywords = None
            summary_length = None
            diagrams = True if request.form.get('diagrams') == 'on' else False
            if input_file and input_file.filename.split('.')[-1] in ['mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'txt']:
                input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
                input_file.save(input_file_path)

                if input_file_path.split('.')[-1] in ['mp4', 'mov', 'avi', 'mkv']:
                    audio_path = input_file_path.replace(input_file_path.split('.')[-1], 'mp3')
                    if not os.path.exists(audio_path):
                        audio_extractor.extract_audio(input_file_path, audio_path)
                    input_file_path = audio_path
                
                if input_file_path.split('.')[-1] in ['mp3', 'wav']:
                    transcript_path = input_file_path.replace('.mp3', '.txt')
                    if not os.path.exists(transcript_path):
                        transcription_result = whisper_transcriber.transcribe(input_file_path)
                        whisper_transcriber.save_transcript(transcription_result, transcript_path)
                    input_file_path = transcript_path
                
                if notes is not None:
                    transcript_processor = TranscriptProcessor(pinecone_api_key=os.getenv('PINECONE_API_KEY'), 
                                                               pinecone_index_name='dl-ai')
                transcript_processor.save_embeddings(input_file_path)

                output = llama.generate_notes(input_file_path, summary_length=500, 
                                                generate_glossary=glossary, generate_takeaways=takeaways)
                notes = format_notes(output)
                chat = [] ##
            return render_template('index.html', notes=notes, chat=chat)
        else:
            question = request.form.get('question')
            if notes is not None:
                context = transcript_processor.query_vector_db(question, 10)
                answer = llama.answer_question(question, context, input_file_path)
            else:
                answer = llama.answer_question(question, None, None)
            chat.append({'name': 'user', 'message': question})
            chat.append({'name': 'llm', 'message': answer})
            return render_template('index.html', notes=notes, chat=chat)
    else:
        return render_template('index.html', notes=notes, chat=chat)


if __name__ == '__main__':
    whisper_transcriber, transcript_processor, audio_extractor, llama = load_models() ##
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, use_reloader=False) ##
    
