from typing import List, Optional
from llama import Dialog, Llama
import json
import time

class TextToNotes:
    def __init__(self, ckpt_dir: str,
                 tokenizer_path: str,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 max_seq_len: int = 8192,
                 max_batch_size: int = 8,
                 max_gen_len: Optional[int] = None,):
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_gen_len = max_gen_len

        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )

    @staticmethod
    def read_text(path: str):
        with open(path, 'r') as file:
            text = file.read()
        return text

    def split_to_chunks(self, text: str, total_words: int):
        # divide text into chunks that are equal in size and do not exceed max_seq_len (in terms of words)
        num_chunks = total_words // self.max_seq_len + 1
        chunk_size = total_words // num_chunks + 1
        chunk_summary_length = self.max_seq_len // num_chunks - 1
        words_in_sentences = [sentence.split() for sentence in text.split('.')]
        
        # chunks should have lesser than chunk_size words and should be complete sentences
        chunks = []
        chunk = ""
        for sentence in words_in_sentences:
            if len(chunk.split()) + len(sentence) < chunk_size:
                chunk += ' '.join(sentence) + '.'
            else:
                chunks.append(chunk)
                chunk = ' '.join(sentence) + '.'
        chunks.append(chunk)

        return chunks, chunk_summary_length

    def generate_chunk_summary(self, text: str, summary_length: int = 250, generate_glossary: bool = False): 
        context_prompt = f"Here's a text transcript of a part of a lecture: \n\n{text}\n\n\
        The timestamp is included at the end of each sentence in the format (timestamp: <TIMESTAMP>)."
        summary_prompt = f"Summarize the transcript to a summary of close to {summary_length} words, but don't exceed it. \
        Make sure to carry over the timestamps in the same format with every sentence in the summary. \
        If there are multiple timestamps for a sentence in the summary, keep the first one."
        glossary_prompt = "Generate a glossary of key terms and their definitions as html list using <ul> and <li> tags. \
            Do not add any introductory text."
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": context_prompt},
                {"role": "user", "content": summary_prompt},
            ]
        ]

        if generate_glossary:
            dialogs.append([
                {"role": "user", "content": context_prompt},
                {"role": "user", "content": glossary_prompt},
            ])
        
        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return results[0]['generation']['content'], (results[1]['generation']['content'] if generate_glossary else None)

    def generate_notes(self, input_path: str, summary_length: int = 250,
                       generate_takeaways: bool = False, generate_glossary: bool = False, 
                       save_path: Optional[str] = None):
        text = self.read_text(input_path)
        total_words = len(text.split())

        dialogs : List[Dialog] = []
        summary_prompt = f"Summarize the lecture to a {summary_length} words (approximate) summary. \
        Make sections with headings when necessary. Include the timestamp along with every heading \
        and with every 5th or 6th sentence in the same format without the word 'timestamp'. Enclose headings in h3 html tags."
        glossary_prompt = "Generate a glossary of key terms and their definitions as html list using <ul> and <li> tags. \
            Do not add any introductory text."
        takeaways_prompt = "Generate a list of 5 to 10 key takeaways from the lecture as html list using <ul> and <li> tags. \
            Do not add any introductory text."
        context_prompt = f"Here's a text transcript of a lecture: \n\n"
        context_prompt_end = "\n\nThe timestamp is included at the end of each sentence in the format (timestamp: <TIMESTAMP>)."

        if total_words > self.max_seq_len:
            print('chunking')
            chunks, chunk_summary_length = self.split_to_chunks(text, total_words)

            # generate summaries and glossaries for each chunk and combine them
            text = ""
            if generate_glossary:
                glossary = ""
            for chunk in chunks:
                chunk_summary, chunk_glossary = self.generate_chunk_summary(chunk, chunk_summary_length, 
                                                                            generate_glossary=generate_glossary)
                text += chunk_summary + " "
                if generate_glossary:
                    glossary += chunk_glossary + "\n"

        else:
            if generate_glossary:
                dialogs.append([
                    {"role": "user", "content": context_prompt + text + context_prompt_end},
                    {"role": "user", "content": glossary_prompt},
                ])
            

        dialogs.append([
            {"role": "user", "content": context_prompt + text + context_prompt_end},
            {"role": "user", "content": summary_prompt},
        ])
        if generate_takeaways:
            dialogs.append([
                {"role": "user", "content": context_prompt + text + context_prompt_end},
                {"role": "user", "content": takeaways_prompt},
            ])
        start_time = time.time()
        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        notes = {"summary": None, "takeaways": None, "glossary": None}
        if total_words > self.max_seq_len:
            if generate_glossary:
                notes["glossary"] = glossary
            notes["summary"] = results.pop(0)['generation']['content']
            if generate_takeaways:
                notes["takeaways"] = results.pop(0)['generation']['content']
        else:
            if generate_glossary:
                notes["glossary"] = results.pop(0)['generation']['content']
            notes["summary"] = results.pop(0)['generation']['content']
            if generate_takeaways:
                notes["takeaways"] = results.pop(0)['generation']['content']
    
        if save_path:
            with open(save_path, 'w') as file:
                json.dump(notes, file)

        return notes

    def answer_question(self, question, context, input_file_path):
        dialogs = [
            [
                {"role": "user", "content": question},
            ]
        ]
        
        if input_file_path:
            text = self.read_text(input_file_path)
        context_prompt = f"Here's the text transcript from a lecture: \n\n"

        if input_file_path and (len(text.split()) < (self.max_seq_len - 100)): 
            dialogs[0].insert(0, {"role": "user", "content": context_prompt + text})
        elif context:
            dialogs[0].insert(0, {"role": "user", "content": context_prompt + ' '.join(context)}) ## 

        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return results[0]['generation']['content']

if __name__ == "__main__":
    ckpt_dir = "../llama3/Meta-Llama-3-8B-Instruct/"
    tokenizer_path = "../llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    text_path = "../transcription_result_with_timestamp.txt"
    text_to_notes = TextToNotes(ckpt_dir, tokenizer_path)
    notes = text_to_notes.generate_notes(text_path, summary_length=500, generate_takeaways=True, 
                                         generate_glossary=True, save_path="../notes.json")