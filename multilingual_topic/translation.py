
import json
from pathlib import Path
from itertools import islice
from typing import List, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformers import (
        MBartForConditionalGeneration,
        MBart50TokenizerFast,
        MarianMTModel,
        MarianTokenizer,
    )
except ImportError:
    pass

# ISO 639-1 → MBART language code
# Full list: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
MBART_LANG_CODES = {
    'af': 'af_ZA', 'ar': 'ar_AR', 'az': 'az_AZ', 'bn': 'bn_IN',
    'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 'es': 'es_XX',
    'et': 'et_EE', 'fa': 'fa_IR', 'fi': 'fi_FI', 'fr': 'fr_XX',
    'gl': 'gl_ES', 'gu': 'gu_IN', 'he': 'he_IL', 'hi': 'hi_IN',
    'hr': 'hr_HR', 'id': 'id_ID', 'it': 'it_IT', 'ja': 'ja_XX',
    'ka': 'ka_GE', 'kk': 'kk_KZ', 'km': 'km_KH', 'ko': 'ko_KR',
    'lt': 'lt_LT', 'lv': 'lv_LV', 'mk': 'mk_MK', 'ml': 'ml_IN',
    'mn': 'mn_MN', 'mr': 'mr_IN', 'my': 'my_MM', 'ne': 'ne_NP',
    'nl': 'nl_XX', 'pl': 'pl_PL', 'ps': 'ps_AF', 'pt': 'pt_XX',
    'ro': 'ro_RO', 'ru': 'ru_RU', 'si': 'si_LK', 'sl': 'sl_SI',
    'sq': 'sq_AL', 'sv': 'sv_SE', 'sw': 'sw_KE', 'ta': 'ta_IN',
    'te': 'te_IN', 'th': 'th_TH', 'tl': 'tl_XX', 'tr': 'tr_TR',
    'uk': 'uk_UA', 'ur': 'ur_PK', 'vi': 'vi_VN', 'xh': 'xh_ZA',
    'zh': 'zh_CN',
}


# ── High-level Translator (primary user-facing class) ────────────────────────

class _MBartDataset(Dataset):
    """Internal dataset for MBART batch tokenisation."""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
        }


class Translator:
    """Batch machine translator using MBART (many-to-many) or Helsinki/Marian models.

    Two model families are supported:

    - **MBART** (``facebook/mbart-large-50-many-to-many-mmt``): translates between
      any pair of its 50 languages. Recommended for most use cases — higher quality,
      especially for longer or domain-specific text.

    - **Helsinki/Marian** (``Helsinki-NLP/opus-mt-{src}-{tgt}``): language-pair-specific
      models, smaller and faster (~2× speedup). Useful when compute is the bottleneck.

    Parameters
    ----------
    src_lang : str
        ISO 639-1 source language code, e.g. ``'ar'``, ``'tr'``, ``'fr'``.
    model_name : str
        HuggingFace model ID. Defaults to MBART many-to-many.
        Pass ``'helsinki'`` as a shorthand for ``Helsinki-NLP/opus-mt-{src}-{tgt}``.
    tgt_lang : str
        ISO 639-1 target language code. Defaults to ``'en'``.
    device : str or None
        ``'cuda'``, ``'cpu'``, or ``None`` to auto-detect.

    Examples
    --------
    Arabic → English with MBART:

    >>> translator = Translator(src_lang='ar')
    >>> translations = translator.translate(['مرحبا بالعالم', 'كيف حالك؟'])

    Turkish → English with Helsinki (faster):

    >>> translator = Translator(src_lang='tr', model_name='helsinki')
    >>> translations = translator.translate(turkish_sentences)
    """

    MBART_MODEL = 'facebook/mbart-large-50-many-to-many-mmt'

    def __init__(
        self,
        src_lang: str,
        model_name: str = 'facebook/mbart-large-50-many-to-many-mmt',
        tgt_lang: str = 'en',
        device: str = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name == 'helsinki':
            model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

        self.model_name = model_name
        self._is_mbart = 'mbart' in model_name.lower()

        if self._is_mbart:
            if src_lang not in MBART_LANG_CODES:
                raise ValueError(
                    f"'{src_lang}' not in MBART language codes. "
                    f"Supported: {sorted(MBART_LANG_CODES)}"
                )
            if tgt_lang not in MBART_LANG_CODES:
                raise ValueError(f"'{tgt_lang}' not in MBART language codes.")
            self._src_code = MBART_LANG_CODES[src_lang]
            self._tgt_code = MBART_LANG_CODES[tgt_lang]
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.tokenizer.src_lang = self._src_code
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self._forced_bos = self.tokenizer.lang_code_to_id[self._tgt_code]
        else:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self._forced_bos = None

        self.model.to(self.device)
        self.model.eval()

    def translate(
        self,
        texts: list,
        batch_size: int = 16,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> list:
        """Translate a list of strings from ``src_lang`` to ``tgt_lang``.

        Parameters
        ----------
        texts : list of str
        batch_size : int
            Sentences per GPU batch. Reduce if you hit OOM.
        max_length : int
            Max tokens for input and output. Longer text is truncated.
        show_progress : bool

        Returns
        -------
        list of str  — translated strings in the same order as ``texts``.
        """
        if self._is_mbart:
            return self._translate_mbart(texts, batch_size, max_length, show_progress)
        return self._translate_marian(texts, batch_size, max_length, show_progress)

    def _translate_mbart(self, texts, batch_size, max_length, show_progress):
        dataset = _MBartDataset(texts, self.tokenizer, max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        out = []
        for batch in tqdm(loader, disable=not show_progress):
            data = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                tokens = self.model.generate(**data, forced_bos_token_id=self._forced_bos)
            out.extend(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
        return out

    def _translate_marian(self, texts, batch_size, max_length, show_progress):
        out = []
        for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            ).to(self.device)
            with torch.no_grad():
                tokens = self.model.generate(**enc)
            out.extend(self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
        return out

    @staticmethod
    def mbart_languages() -> dict:
        """Return the ISO 639-1 → MBART language code mapping."""
        return dict(MBART_LANG_CODES)

    def __repr__(self):
        return (
            f"Translator(src='{self.src_lang}', tgt='{self.tgt_lang}', "
            f"model='{self.model_name}', device='{self.device}')"
        )


# ── Low-level / streaming classes (used by translate_iterator) ───────────────

class _BaseTranslator:
    """Base class for low-level translators that require an externally loaded model."""
    def translate(self, texts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError


class ManyToManyTranslator(_BaseTranslator):
    """Low-level wrapper for MBART many-to-many, expects pre-loaded model/tokenizer."""

    def __init__(self, name: str, model, tokenizer, max_length, use_gpu) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gpu = use_gpu

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            texts, return_tensors='pt', padding='max_length',
            truncation=True, max_length=self.max_length,
        )
        if self.use_gpu:
            encoded = encoded.to('cuda')
        generated_tokens = self.model.generate(
            **encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


class HelsinkiTranslator(_BaseTranslator):
    """Low-level wrapper for Helsinki/Marian models, expects pre-loaded model/tokenizer."""

    def __init__(self, name: str, model, tokenizer, max_length, use_gpu) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gpu = use_gpu

    def translate(self, texts: List[str], **kwargs) -> List[str]:
        encoded = self.tokenizer(
            texts, return_tensors='pt', padding='max_length',
            truncation=True, max_length=self.max_length,
        )
        if self.use_gpu:
            encoded = encoded.to('cuda')
        generated_tokens = self.model.generate(**encoded)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]


# ── Streaming / incremental translation utilities ────────────────────────────

def iter_batches(gen: Iterable, batch_size: int):
    """Split a generator into fixed-size chunks."""
    gen = iter(gen)
    chunk = lambda: list(islice(gen, batch_size))
    return iter(chunk, [])


def iter_df_as_dict(df):
    for _, row in df.iterrows():
        yield dict(row)


def translate_iterator(
    dict_iter,
    batch_size: int,
    translator,
    text_col: str,
    src_lang: str,
    tgt_lang: str,
    translated_col: str,
    translated_by_col: str,
):
    """Translate a dict iterator in batches, yielding augmented dicts.

    Useful for streaming large datasets without loading everything into memory.
    """
    for data_batch in iter_batches(dict_iter, batch_size=batch_size):
        text_batch = [d[text_col] for d in data_batch]
        translated_text_batch = translator.translate(text_batch, src_lang=src_lang, tgt_lang=tgt_lang)
        for d, t in zip(data_batch, translated_text_batch):
            d[translated_col] = t
            d[translated_by_col] = translator.name
            yield d


def get_processed_message_ids(jsonl_path: str, id_col: str) -> set:
    """Return the set of IDs already written to a JSONL file.

    Used to resume an interrupted translation run without reprocessing.
    """
    seen_ids = set()
    if not Path(jsonl_path).exists():
        Path(jsonl_path).touch()
    with open(jsonl_path) as f:
        for line in f:
            seen_ids.add(json.loads(line)[id_col])
    return seen_ids
