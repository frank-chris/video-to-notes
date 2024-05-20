# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
import re

# %%
# get api key for pinecone
PINECONE_API_KEY = 'fa4f8bdf-560e-4992-bcf5-1e7a7269b410'


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')


# %%


class TranscriptProcessor:
    def __init__(self, pinecone_api_key, pinecone_index_name, model_name='all-MiniLM-L6-v2'):
        # Initialize Pinecone
        pinecone = Pinecone(api_key=pinecone_api_key)
        if pinecone_index_name in [index.name for index in pinecone.list_indexes()]:
            pinecone.delete_index(pinecone_index_name)
        self.model = SentenceTransformer(model_name, device=device)
        pinecone.create_index(name=pinecone_index_name, dimension=self.model.get_sentence_embedding_dimension(), metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
        self.index = pinecone.Index(pinecone_index_name)
        
        # Initialize SentenceTransformer model
        

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

    def save_embeddings(self, chunks):
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




# %%
# Example usage
processor = TranscriptProcessor(
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_index_name='dl-ai'
)

# Step 1: Read the transcript
transcript = processor.read_transcript('transcription_result_with_timestamp.txt')

# Step 2: Chunk the transcript by sentences
chunks = processor.chunk_transcript(transcript, chunk_size=50)

# Step 3: Save embeddings to Pinecone
processor.save_embeddings(chunks)


# %%
# Step 4: Query vector DB with a question
context = processor.query_vector_db("what is BBC implementation", 5)



