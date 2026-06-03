
import json
from pathlib import Path
from itertools import islice
from typing import List, Iterable


class Translator:
    """
    Base class for translators.
    """
    def translate(self, texts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError


class ManyToManyTranslator(Translator):
    """
    Class for translators that can translate many-to-many, e.g. mbart-large-50-many-to-many-mmt.
    """
    def __init__(self, name: str, model, tokenizer, max_length, use_gpu) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gpu = use_gpu

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        self.tokenizer.src_lang = src_lang

        encoded = self.tokenizer(texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )

        if self.use_gpu:
            encoded = encoded.to("cuda")

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang]
        )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


class HelsinkiTranslator(Translator):
    """
    Translator for one-to-one Helsinki-NLP models.
    """
    def __init__(self, name: str, model, tokenizer, max_length, use_gpu) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gpu = use_gpu


    def translate(self, texts: List[str], **kwargs) -> List[str]:
        encoded = self.tokenizer(texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )
        if self.use_gpu:
            encoded = encoded.to("cuda")

        generated_tokens = self.model.generate(**encoded)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]


def iter_batches(gen: Iterable, batch_size):
    """ Read parts of the generator, pause each time after a chunk """
    gen = iter(gen)
    make_chunk = lambda: list(islice(gen, batch_size))
    return iter(make_chunk, [])

def iter_df_as_dict(df):
    for _, row in df.iterrows():
        yield dict(row)

def translate_iterator(dict_iter, batch_size: int, translator, text_col: str,
                       src_lang: str, tgt_lang: str, translated_col: str, translated_by_col: str):
    for data_batch in iter_batches(dict_iter, batch_size=batch_size):
        text_batch = [d[text_col] for d in data_batch]
        translated_text_batch = translator.translate(text_batch, src_lang=src_lang, tgt_lang=tgt_lang)

        for d, t in zip(data_batch, translated_text_batch):
            d[translated_col] = t
            d[translated_by_col] = translator.name
            yield d

def get_processed_message_ids(jsonl_path, id_col):

    seen_ids = set()

    if not Path(jsonl_path).exists():
        Path(jsonl_path).touch()

    with open(jsonl_path) as f:
        for line in f:
            msg_id = json.loads(line)[id_col]
            seen_ids.add(msg_id)

    return seen_ids


# class TranslationDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         article = self.data[idx]
#         encoded = self.tokenizer(
#             article, return_tensors="pt", padding="max_length",
#             truncation=True, max_length=self.max_length
#         )
#         return {
#             'input_ids': encoded['input_ids'].squeeze(),
#             'attention_mask': encoded['attention_mask'].squeeze()
#         }
#
#
# class Translation:
#     def __init__(self, model_name, dataset, text_col='M52/text', max_length=512, batch_size=16):
#         self.model_name = model_name
#         self.dataset = dataset
#         self.text_col = text_col
#         self.max_length = max_length
#         self.batch_size = batch_size
#
#
#     def get_model(self):
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
#         return device, tokenizer, model
#
#     def generate_dataset(self, tokenizer):
#         translation_dataset = TranslationDataset(self.dataset[self.text_col].tolist(), tokenizer, max_length=512)
#         dataloader = DataLoader(translation_dataset, batch_size=self.batch_size, shuffle=False)
#         return dataloader
#     def translate(self):
#         device, tokenizer, model = self.get_model()
#         dataloader = self.generate_dataset(tokenizer)
#         model.to(device)
#         translations = []
#
#         for batch in tqdm(dataloader):
#             data = {k: v.to('cuda') for k, v in batch.items()}
#
#             generated_tokens = model.generate(
#                 **data
#             )
#             batch_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#             translations.extend(batch_translation)
#
#         self.dataset['translations'] = translations
#
#         return self.dataset
