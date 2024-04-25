from typing import List, Optional
from llama import Dialog, Llama


class TextToNotes:
    def __init__(self, ckpt_dir: str,
                 tokenizer_path: str,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 max_seq_len: int = 8192,
                 max_batch_size: int = 6,
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

    def generate_summary_chunk(self, text: str, summary_length: int = 200): 
        context_prompt = f"Here's a text transcript of a lecture: \n\n{text}"
        summary_prompt = f"Summarize the lecture to a {summary_length} words (approximate) paragraph."
        dialogs: List[Dialog] = [
            [
                {"role": "user", "content": context_prompt},
                {"role": "user", "content": summary_prompt},
            ]
        ]
        
        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        return results[0]['generation']['content']

    def generate_summary(self, input_path: str, save_path: Optional[str] = None):
        text = self.read_text(input_path).split()
        total_words = len(text)

        # divide text into chunks that are equal in size and do not exceed max_seq_len (in terms of words)
        num_chunks = total_words // self.max_seq_len + 1
        chunk_size = total_words // num_chunks + 1
        chunks = [text[i:i + chunk_size] for i in range(0, total_words, chunk_size)]

        # generate summaries for each chunk 
        summaries = []  
        for chunk in chunks:
            text = ' '.join(chunk)
            summary = self.generate_summary_chunk(text)
            summaries.append(summary)

        # combine summaries into one
        summary = ' '.join(summaries)

        if save_path:
            with open(save_path, 'w') as file:
                file.write(summary)

        return summary


if __name__ == "__main__":
    ckpt_dir = "../llama3/Meta-Llama-3-8B-Instruct/"
    tokenizer_path = "../llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    text_path = "../engraft.txt"
    text_to_notes = TextToNotes(ckpt_dir, tokenizer_path)
    summary = text_to_notes.generate_summary(text_path, save_path="../summary.txt") 
    print(summary)