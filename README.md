# multilingual-topic-toolkit

A Python toolkit for multilingual topic modelling, SetFit-guided cluster refinement, machine translation, and text preprocessing. Built for large-scale social media and news analysis across multiple language communities.

## Modules

| Module | Description |
|---|---|
| `topic_modeling` | BERTopic-based topic modelling with UMAP + HDBSCAN; sentence embedding extraction; hyperparameter tuning |
| `classification` | SetFit few-shot classifier helpers for contrastive cluster refinement, validation and annotation |
| `translation` | Batched machine translation pipeline supporting mBART (many-to-many) and Helsinki-NLP models |
| `preprocessing` | Multilingual text preprocessor — links, mentions, hashtags, emojis, Arabic normalisation |
| `checks` | Tokenisation truncation checks, language-aware sentence splitters, text chunker with validation |
| `utils` | File I/O utilities (CSV, JSON, JSONL, TMX, parallel corpora) |

## Installation

```bash
pip install -e .
```

For Arabic preprocessing support:

```bash
pip install -e ".[arabic]"
```

## Architecture

This toolkit was developed to support a **layered topic modelling** approach for multilingual corpora where clusters are dense and semantically entangled.

### The problem
Standard single-pass topic models often produce clusters that are too broad or too noisy to be analytically useful — especially on multilingual social media where a single theme (e.g. "child welfare") generates very different surface forms across languages.

### The approach

```
Layer 1  →  Global BERTopic across the full corpus
             Broad thematic clusters identified

Layer 2  →  Per-theme BERTopic models
             Sub-topics discovered within each cluster
             SetFit used to refine boundaries

Layer 3  →  Deep topic model on specific dense narratives
             Further unpacked using SetFit contrastive learning
```

### Why SetFit?
SetFit (few-shot fine-tuning of sentence transformers) is used **between topic model layers** to steer the semantic structure of clusters. Rather than accepting raw cluster output, SetFit is trained on positive and negative examples selected from the cluster — teaching the model what makes a good vs. bad grouping. This shapes the embedding space used by the next BERTopic layer, producing more coherent and analytically meaningful topics with minimal labelled data.

## Quick start

### Topic modelling

```python
from multilingual_topic import SentenceRepresentation, TopicModel

# Embed documents
embedder = SentenceRepresentation(
    model_name="all-mpnet-base-v2",
    data=df,
    text_col="text"
)
embedder.extract_embeddings(batch_size=64)

# Fit topic model
model = TopicModel(data=df, text_col="text")
topics, probs = model.fit(embedder.embeddings)
```

### SetFit cluster refinement

```python
from multilingual_topic import ClassifierValidationHelper
from setfit import SetFitModel

# Prepare positive/negative examples for a target cluster
helper = ClassifierValidationHelper(
    data=cluster_df,
    theme_col="cluster_label",
    theme="target_cluster",
    src_text="text",
    text_col="text_clean",
    label_col="label",
    prediction_col="pred",
    topic_col="topic_id",
    sample_size=8,
)

setfit_model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
train_dataset, valid_dataset = helper.convert_split_to_dataset(
    *helper.split_data(helper.prepare_data()), sample_size=8
)
# Train setfit_model on train_dataset, then use refined embeddings for next BERTopic layer
```

### Machine translation

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from multilingual_topic import ManyToManyTranslator, translate_iterator
from multilingual_topic.translation import iter_df_as_dict

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

translator = ManyToManyTranslator("mbart", model, tokenizer, max_length=512, use_gpu=True)

translated = list(translate_iterator(
    iter_df_as_dict(df), batch_size=16, translator=translator,
    text_col="text", src_lang="ar_AR", tgt_lang="en_XX",
    translated_col="text_en", translated_by_col="translator"
))
```

### Preprocessing

```python
from multilingual_topic import TextPreprocessor

pp = TextPreprocessor(
    apply_remove_links=True,
    apply_remove_mentions=True,
    apply_remove_hashtags=True,
    apply_remove_emojis=True,
    apply_arabic_preprocessor=False,  # set True for Arabic with AraBERT
)
df["text_clean"] = df["text"].apply(pp.preprocess)
```

### Sentence splitting and chunking

```python
from multilingual_topic import SentenceSplitterGenerator, TextChunker

splitter = SentenceSplitterGenerator.get_splitter("ar")  # Arabic
sentences = splitter.simple_split(text)

chunker = TextChunker(tokenizer)
chunks = chunker.create_chunks(sentences)
```

## Languages supported

| Language | Preprocessing | Sentence splitting | Translation |
|---|---|---|---|
| Arabic | ✓ (+ AraBERT) | ✓ | ✓ (mBART / Helsinki) |
| English | ✓ | ✓ | ✓ |
| Urdu | ✓ | ✓ | ✓ |
| Turkish | ✓ | ✓ | ✓ |
| Indonesian | ✓ | ✓ | ✓ |
| Hindi | ✓ | ✓ | ✓ |
| Persian/Farsi | ✓ | — | ✓ |
