import re
import math
import pandas as pd
from typing import List
from pathlib import Path
import plotly.express as px
from tqdm.notebook import tqdm


class TruncationChecks:
    """
    Class for checking tokenization truncation in data.
    """

    def __init__(self, tokenizer, max_length, data, text_col='text', id_col='message_id',
                 batch_size=64):
        self.tokenizer = tokenizer
        self.data = data
        self.text_col = text_col
        self.id_col = id_col
        self.batch_size = batch_size
        self.tokenizer.model_max_length = max_length
        self.truncation_df = pd.DataFrame()  # Initialize as empty DataFrame
        print(f'Tokenizer Max Length: {self.tokenizer.model_max_length}')
        self.tokens = self._extract_tokens()

    def _extract_tokens(self):
        """
        Tokenizes the data in batches and stores the tokens.
        """
        data_list = self.data[self.text_col].tolist()

        total_batches = math.ceil(len(data_list) / self.batch_size)
        tokens = []

        for i in tqdm(range(total_batches), desc="Tokenizing"):
            batch = data_list[i * self.batch_size: (i + 1) * self.batch_size]
            batch_tokens = self.tokenizer.batch_encode_plus(batch, truncation=False)
            tokens.extend(batch_tokens['input_ids'])
        return tokens

    def check_for_truncation(self, directory: str):
        """
        Checks each tokenized input for truncation and returns a DataFrame with details.
        """
        file_path = Path(directory) / f'truncation_df.csv'
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        tokenized_data = []
        for message_id, text, token_ids in tqdm(zip(self.data[self.id_col], self.data[self.text_col], self.tokens),
                                                desc="Truncation Checks"):
            tokenization_length = len(token_ids) - 2
            truncated = tokenization_length > self.tokenizer.model_max_length
            truncation_amount = max(0, tokenization_length - self.tokenizer.model_max_length)

            tokenized_text, truncated_text = '', ''
            if truncated:
                tokenized_text = self.tokenizer.decode(token_ids[:self.tokenizer.model_max_length],
                                                       skip_special_tokens=True)
                truncated_text = self.tokenizer.decode(token_ids[self.tokenizer.model_max_length:],
                                                       skip_special_tokens=True)

            tokenized_data.append({
                self.id_col: message_id,
                self.text_col: text,
                "tokenized_text": tokenized_text,
                "tokenization_length": tokenization_length,
                "truncated": truncated,
                "truncation_amount": truncation_amount,
                "truncated_text": truncated_text,
            })
        self.truncation_df = pd.DataFrame(tokenized_data)
        self.truncation_df.to_csv(
            file_path,
            index=False
        )

    def trucation_amount_distribution(self, directory: str, max_amount=None, log_scale=False):

        self.check_for_truncation(directory)
        data = self.truncation_df.copy()

        print(f"Number of not truncated data: {len(data[~data['truncated']])}")
        print()
        print(f"Number of truncated data: {len(data[data['truncated']])}")
        print()
        print(f"Maximum truncation amount: {data['truncation_amount'].max()}")
        print()
        fig = px.histogram(data, x='truncation_amount', title='Truncation Amount Distribution')

        fig.update_layout(
            xaxis_title='Truncation Amount',
            yaxis_title='Frequency',
            xaxis=dict(range=[0, max_amount] if max_amount is not None else None),
            yaxis=dict(type='log' if log_scale else 'linear')
        )
        fig.add_vline(x=self.tokenizer.model_max_length, line_width=3, line_dash="dash", line_color="red")
        fig.write_image(directory / "truncation_distribution.png")
        fig.show()
        return fig


class SentenceSplitter:
    """
    Base class for sentence splitters.
    """

    def simple_split(self, text: str) -> List[str]:
        raise NotImplementedError

    def check_splitter_output(self):
        print(f'Number of sentences: {len(self.sentences)}')
        for sentence in self.sentences:
            print(sentence)


class EnglishSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for English.
    """

    def simple_split(self, text: str) -> List[str]:
        """
        Ignore e.g.
        Ignore Dr.
        Only split when it ends with . or ?, then white space
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences


class ArabicSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for Arabic.
    """

    def simple_split(self, text: str) -> List[str]:
        """

        Only split when it ends with . or ؟
        """
        sentences = re.split(r'(?<=\u061F|\u002E)\s', text)
        return sentences


class IndonesianSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for Arabic.
    """

    def simple_split(self, text: str) -> List[str]:
        """

        Only split when it ends with . or ؟
        """
        sentences = re.split(r'(?<=[.!?])\s', text)
        return sentences


class TurkishSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for Arabic.
    """

    def simple_split(self, text: str) -> List[str]:
        """

        Only split when it ends with . or ؟
        """
        sentences = re.split(r'(?<=[.!?])\s', text)
        return sentences


class HindiSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for Arabic.
    """

    def simple_split(self, text: str) -> List[str]:
        """

        Only split when it ends with . or ؟
        """
        sentences = re.split(r'(?<=[।.!?])\s+', text)
        return sentences


class UrduSentenceSplitter(SentenceSplitter):
    """
    Sentence splitter for Arabic.
    """

    def simple_split(self, text: str) -> List[str]:
        """

        Only split when it ends with . or ؟
        """
        sentences = re.split(r'(?<=[۔.!?])\s+', text)
        return sentences


# ... and so on for other languages

class SentenceSplitterGenerator:
    """
    Factory class to get the appropriate sentence splitter for a given language.
    """
    splitters = {
        'en': EnglishSentenceSplitter(),
        'ar': ArabicSentenceSplitter(),
        'id': IndonesianSentenceSplitter(),
        'tr': TurkishSentenceSplitter(),
        'hi': HindiSentenceSplitter(),
        'ur': UrduSentenceSplitter(),

        # Add other language-specific splitters
    }

    @staticmethod
    def get_splitter(lang_code: str) -> SentenceSplitter:
        if lang_code in SentenceSplitterGenerator.splitters:
            return SentenceSplitterGenerator.splitters[lang_code]
        raise NotImplementedError(f"No sentence splitter for language code: {lang_code}")


class TextChunker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length

    def create_chunks(self, sentences):
        chunks = []
        current_chunk = []
        chunk_id = 0
        for sentence in sentences:
            new_tokens = self.tokenizer.encode(' '.join(current_chunk + [sentence]))
            # If the new chunk is too long and the current chunk is empty (very long sentence)
            if len(new_tokens) - 2 > self.max_seq_length and not current_chunk:
                # Split the sentence further or truncate, then continue (adjust as needed)
                current_chunk = [sentence]  # Simple truncation example
                continue
            # If the new chunk is within the limit, add the sentence to the current chunk
            if len(new_tokens) <= self.max_seq_length:
                current_chunk.append(sentence)
            else:
                # If adding the sentence exceeds the limit, save the current chunk and start a new one
                chunks.append((' '.join(current_chunk), chunk_id))
                current_chunk = [sentence]
                chunk_id += 1

        # Don't forget to add the last chunk
        if current_chunk:
            chunks.append((' '.join(current_chunk), chunk_id))
        return chunks


class BaseTester:
    def assertEqual(self, actual, expected, id, msg=None):
        if actual != expected:
            raise AssertionError(msg or f"Expected {expected}, got {actual} in {id}")

    def assertTrue(self, expr, id, msg=None):
        if not expr:
            raise AssertionError(msg or f"Expected True but got False in {id}")

    # You can add more assertion methods as needed, similar to those in unittest.TestCase


class TestChunker(BaseTester):

    def __init__(self, id, text, sentences, chunked_data, tokenizer, splitter):
        # Assume tokenizer and max_seq_length are properly defined
        self.id = id
        self.text = text
        self.sentences = sentences
        self.chunked_data = chunked_data
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length
        self.splitter = splitter

    def test_no_sentence_loss(self):
        flattened_chunked_sentences = [sentence for chunk in self.chunked_data for sentence in
                                       self.splitter.simple_split(chunk[0])]
        try:
            self.assertEqual(len(self.sentences), len(flattened_chunked_sentences), self.id)
        except AssertionError as e:
            print(f"test_no_sentence_loss failed: {e}")
            return self.id

    def test_chunk_size_validation(self):
        for chunk in self.chunked_data:
            try:
                self.assertTrue(len(self.tokenizer.encode(chunk[0])) <= self.max_seq_length, (self.id, chunk[1]))
            except AssertionError as e:
                print(f"test_chunk_size_validation failed: {e}")
                return self.id

    def test_edge_cases(self):
        try:
            self.assertEqual(len(self.chunked_data), 1, self.id)
        except AssertionError as e:
            print(f"test_edge_cases failed: {e}")
            return self.id

    def test_correct_chunking(self):
        all_chunked_text = " ".join([chunk[0] for chunk in self.chunked_data])
        try:
            self.assertEqual(self.text.replace("\n", "").replace(" ", ""), all_chunked_text.replace(" ", ""), self.id)
        except AssertionError as e:
            print(f"test_correct_chunking failed: {e}")
            return self.id

    def run_tests(self):
        return {
            '#_sentences': self.test_no_sentence_loss(),
            'chunk_size': self.test_chunk_size_validation(),
            'correct_chunking': self.test_correct_chunking()
        }


def identify_failiures(fialiures):
    article_ids = []
    for id, failiure in enumerate(fialiures):
        for k, v in failiure.items():
            if v != None:
                print(id, k, v)
                article_ids.append(v)
    return article_ids


def debug_chunk(truncated_data, failed_cases, id_col='articleId'):
    article_ids = identify_failiures(failed_cases)
    return [truncated_data[truncated_data[id_col] == article_id]['text'].iloc[0] for article_id in article_ids]


def check_text(text, fialiure_id, tokenizer, splitter):
    split_sentences = splitter.simple_split(text)
    sentences = [sentence.replace('\n', ' ') for sentence in split_sentences]

    # Initialize the chunker
    chunker = TextChunker(tokenizer)

    # Create chunks for the article
    chunks = chunker.create_chunks(sentences)
    print(f'Test Case: {fialiure_id} Length of Sentences: {len(sentences)}, Length of Chunks: {len(chunks)}')
    return sentences, chunks


def fine_chunks(chunks, tokenizer):
    for i, chunk in enumerate(chunks):
        if len(tokenizer.tokenize(chunk[0])) > 1024:
            print(i)
