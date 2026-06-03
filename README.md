# multilingual-topic-toolkit

A Python toolkit for multilingual topic modelling, machine translation, SetFit-guided cluster refinement, and text preprocessing. Built for large-scale social media and news analysis across multiple language communities.

## Modules

| Module | Description |
|---|---|
| `topic_modeling` | BERTopic-based topic modelling with UMAP + HDBSCAN; sentence embedding extraction |
| `translation` | Batched machine translation pipeline supporting mBART (many-to-many) and Helsinki-NLP models |
| `classification` | SetFit few-shot classifier helpers for validation, testing and contrastive cluster refinement |
| `preprocessing` | Multilingual text preprocessor — links, mentions, hashtags, emojis, Arabic normalisation |
| `checks` | Tokenisation truncation checks, language-aware sentence splitters, text chunker with validation |

## Installation

```bash
pip install -e .
```

For Arabic preprocessing support:

```bash
pip install -e ".[arabic]"
```

## Quick start

### Topic modelling

```python
from multilingual_topic import BERTopicModel, SentenceRepresentation

# Embed and cluster
embedder = SentenceRepresentation(model_name="all-mpnet-base-v2", data=df, text_col="text")
embedder.extract_embeddings(batch_size=64)

model = BERTopicModel(embeddings=embedder.embeddings, documents=embedder.sentences)
topics, probs = model.fit()
```

### Machine translation

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from multilingual_topic import ManyToManyTranslator, translate_iterator

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

translator = ManyToManyTranslator("mbart", model, tokenizer, max_length=512, use_gpu=True)

translated = list(translate_iterator(
    iter_df_as_dict(df), batch_size=16, translator=translator,
    text_col="text", src_lang="ar_AR", tgt_lang="en_XX",
    translated_col="text_en", translated_by_col="translator"
))
```

### SetFit classification

```python
from multilingual_topic import ClassifierValidationHelper

helper = ClassifierValidationHelper(
    data=df, theme_col="theme", theme="target_theme",
    src_text="text", text_col="text_clean",
    label_col="label", prediction_col="pred", topic_col="topic"
)
train_dataset, valid_dataset = helper.convert_split_to_dataset(train_df, valid_df, sample_size=8)
```

### Preprocessing

```python
from multilingual_topic import TextPreprocessor

pp = TextPreprocessor(apply_arabic_preprocessor=False)
df["text_clean"] = df["text"].apply(pp.preprocess)
```

## Architecture

The toolkit was developed to support a layered topic modelling approach for multilingual corpora:

1. **Layer 1** — global BERTopic model across the full dataset to identify broad themes
2. **Layer 2** — per-theme BERTopic models for sub-topic discovery, guided by SetFit contrastive learning
3. **Layer 3** — deep topic models for narratives with dense cluster structures

SetFit is used between layers to provide positive/negative examples that steer the semantic structure of topic clusters — enabling analytically meaningful groupings without large labelled datasets.
