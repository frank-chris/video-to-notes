from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
import re
import os
from dotenv import load_dotenv

class TranscriptProcessor:
    def __init__(self, pinecone_api_key, pinecone_index_name, model_name='all-MiniLM-L6-v2'):
        # Initialize Pinecone
        pinecone = Pinecone(api_key=pinecone_api_key)
        if pinecone_index_name in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(pinecone_index_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print('Sorry no cuda.')
        self.model = SentenceTransformer(model_name, device=device)
        pinecone.create_index(name=pinecone_index_name, dimension=self.model.get_sentence_embedding_dimension(), metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        self.index = pinecone.Index(pinecone_index_name)
        
    def read_transcript(self, file_path):
        with open(file_path, 'r') as file:
            transcript = file.read()
        return transcript

    def chunk_transcript(self, transcript, chunk_size=50):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcript)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def save_embeddings(self, file_path):
        transcript = self.read_transcript(file_path)
        chunks = self.chunk_transcript(transcript, chunk_size=50)

        for i, chunk in enumerate(tqdm(chunks, desc="Saving embeddings to Pinecone")):
            embedding = self.get_embeddings(chunk)
            self.index.upsert([(str(i), embedding, {"text": chunk})])
    
    def get_embeddings(self, text):
        return self.model.encode(text).tolist()

    def query_vector_db(self, query, top_k):
        # Get embeddings for the query
        query_embedding = self.get_embeddings(query)
        
        # Query Pinecone
        result = self.index.query(top_k=top_k, vector=query_embedding, include_metadata=True, include_values=False)
        
        # Extract relevant context from the result
        context = [match['metadata']['text'] for match in result['matches']]
        return context


if __name__ == '__main__':
    load_dotenv()
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    processor = TranscriptProcessor(
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index_name='dl-ai'
    )

    processor.save_embeddings('../transcription_result_with_timestamp.txt')

    context = processor.query_vector_db("how does it help an artist", 5)

    print(len(context))

    for i, c in enumerate(context):
        print(i, ":", c)


