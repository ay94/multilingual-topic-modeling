# contrastive-topic-modeling

A Python toolkit for multilingual topic modelling, SetFit-guided cluster refinement, machine translation, and text preprocessing. Built for large-scale social media and news analysis across multiple language communities.

## Modules

| Module | Class / Function | Description |
|---|---|---|
| `topic_modeling` | `SentenceRepresentation` | Base class for batched sentence embedding extraction |
| | `NomicRepresentation` | Embedding with nomic-embed-text (adds `clustering:` prefix) |
| | `GeneralRepresentation` | Embedding with any standard sentence-transformer model |
| | `ParameterTuner` | Interactive UMAP + HDBSCAN tuning with silhouette score, cluster scatter, save/load |
| | `TopicModel` | BERTopic wrapper with UMAP + HDBSCAN |
| `setfit` | `train_setfit()` | Fine-tune a SetFit model on few-shot positive/negative cluster examples |
| | `cross_validate_setfit()` | K-fold cross-validation for a SetFit binary classifier |
| | `Evaluation` | Classification report, confusion matrix, per-class metrics |
| | `ClassifierValidationHelper` | Prepare and sample SetFit training/validation datasets from cluster output |
| | `ClassifierAnnotationHelper` | Annotate full cluster with a trained SetFit model |
| | `ClassifierTestingHelper` | Run inference on unlabelled cluster data |
| `translation` | `ManyToManyTranslator` | Batched translation using mBART-family models |
| | `HelsinkiTranslator` | Batched translation using Helsinki-NLP language-pair models |
| | `translate_iterator` | Streaming translation pipeline over a DataFrame iterator |
| `preprocessing` | `TextPreprocessor` | Remove links, mentions, hashtags, emojis; Arabic normalisation via AraBERT |
| `checks` | `TruncationChecks` | Detect and report tokenisation truncation |
| | `SentenceSplitterGenerator` | Language-aware sentence splitter factory (en, ar, tr, id, hi, ur) |
| | `TextChunker` | Split documents into token-safe chunks for long-document processing |
| `analysis` | `volume_over_time()` | Monthly post volume, optionally split by category |
| | `top_n_distribution()` | Horizontal bar chart of top-N categorical values |
| | `platform_language_breakdown()` | Platform × language cross-tabulation |
| | `Heatmap` | Pivot-table heatmap for two categorical dimensions (e.g. country × theme) |
| | `top_accounts_by_country()` | Top-N account activity cross-tabulated against countries |
| `utils` | `FileHandler` | File I/O utilities: CSV, JSON, JSONL, TMX, parallel corpora |

## Installation

```bash
pip install -e .
```

For Arabic preprocessing support:

```bash
pip install -e ".[arabic]"
```

## Architecture

This toolkit was developed to support a **layered topic modelling** approach for multilingual corpora where single-pass topic models produce clusters that are too broad or semantically entangled to be analytically useful.

```
Step 1  Preprocess
        TextPreprocessor → remove noise (links, mentions, hashtags, emojis)

Step 2  Translate
        ManyToManyTranslator / HelsinkiTranslator → translate to English
        Quality validated with BERTScore

Step 3  Embed
        SentenceRepresentation / NomicRepresentation / GeneralRepresentation
        Tested: all-mpnet-base-v2 · nomic-embed-text-v1 · UAE-Large-V1 · paraphrase-mpnet-base-v2

Step 4  Tune
        ParameterTuner → tune UMAP (n_neighbors, n_components, min_dist)
                       → tune HDBSCAN (min_cluster_size, cluster_selection_epsilon)
                       → visualise clusters + silhouette score
                       → save / load configurations

Step 5  Layer 1 Topic Model
        TopicModel (BERTopic + UMAP + HDBSCAN) → broad themes across full corpus

Step 6  Layer 2 Topic Model + SetFit refinement
        For each Layer 1 theme:
          → Sub-corpus BERTopic model
          → ClassifierValidationHelper → sample positive / negative cluster examples
          → train_setfit() → fine-tune SetFit on few-shot examples
          → Refined embeddings steer next BERTopic layer

Step 7  Layer 3 (optional deep dive)
        Repeat Layer 2 on the densest / most complex cluster

Step 8  Evaluate
        Evaluation → classification_report · confusion_matrix · metrics dict
        cross_validate_setfit() → K-fold CV, per-fold F1 and summary stats

Step 9  Analyse
        volume_over_time · Heatmap · top_n_distribution · top_accounts_by_country
```

### Why SetFit between topic model layers?

SetFit (few-shot fine-tuning of sentence transformers) is trained on **positive and negative examples drawn from cluster output** — teaching the model what makes an analytically useful grouping vs. noise. The refined embedding space is then used as input to the next BERTopic layer, producing more coherent sub-topics without large labelled datasets.

Real-world performance (10-fold CV, nomic-embed-text-v1):

| Metric | Mean | Min | Max |
|---|---|---|---|
| Accuracy | 0.861 | 0.816 | 0.931 |
| Precision | 0.857 | 0.828 | 0.924 |
| Recall | 0.855 | 0.816 | 0.928 |
| F1 | 0.854 | 0.818 | 0.926 |

Human validation agreement: **0.90 accuracy**.

## Quick start

See [`notebooks/pipeline_demo.ipynb`](notebooks/pipeline_demo.ipynb) for a full end-to-end walkthrough (Steps 1–9) using synthetic data.

See [`notebooks/analysis_demo.ipynb`](notebooks/analysis_demo.ipynb) for aggregation and visualisation examples.

### Embed + tune + topic model

```python
from contrastive_topic import SentenceRepresentation, ParameterTuner, TopicModel

embedder = SentenceRepresentation(model_name="all-mpnet-base-v2", data=df, text_col="text")
embedder.extract_embeddings(batch_size=64)

tuner = ParameterTuner(embeddings=embedder.embeddings, save_directory="./params")
tuner.tune_umap(n_neighbors=15, n_components=5, min_dist=0.0)
tuner.tune_hdbscan(min_cluster_size=10, cluster_selection_epsilon=0.1)
tuner.apply_parameters()
tuner.visualize_parameters()   # prints silhouette score, shows scatter
tuner.save_parameters("layer1")

model = TopicModel(data=df, text_col="text")
topics, probs = model.fit(embedder.embeddings)
```

### SetFit cluster refinement

```python
from contrastive_topic import ClassifierValidationHelper, train_setfit, Evaluation

helper = ClassifierValidationHelper(
    data=cluster_df, theme_col="label", theme="relevant",
    src_text="text", text_col="text", label_col="label",
    prediction_col="pred", topic_col="topic_id", sample_size=8,
)
train_ds, valid_ds = helper.convert_split_to_dataset(
    *helper.split_data(helper.prepare_data()), sample_size=8
)
model = train_setfit(train_ds, valid_ds, model_name="all-mpnet-base-v2")

ev = Evaluation(valid_ds["label"], model(valid_ds["text"]))
ev.report()
ev.confusion_matrix().show()
```

### Cross-validation

```python
from contrastive_topic import cross_validate_setfit

results = cross_validate_setfit(
    df, text_col="text", label_col="label",
    model_name="nomic-ai/nomic-embed-text-v1",
    n_folds=10, batch_size=16, num_epochs=1,
)
```

### Machine translation

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from contrastive_topic import ManyToManyTranslator, translate_iterator
from contrastive_topic.translation import iter_df_as_dict

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translator = ManyToManyTranslator("mbart", model, tokenizer, max_length=512, use_gpu=True)

translated = list(translate_iterator(
    iter_df_as_dict(df), batch_size=16, translator=translator,
    text_col="text", src_lang="ar_AR", tgt_lang="en_XX",
    translated_col="text_en", translated_by_col="translator",
))
```

### Preprocessing

```python
from contrastive_topic import TextPreprocessor

pp = TextPreprocessor(apply_arabic_preprocessor=False)
df["text_clean"] = df["text"].apply(pp.preprocess)
```

### Analysis

```python
from contrastive_topic import volume_over_time, Heatmap, top_n_distribution

volume_over_time(df, date_col="date", group_col="country", top_n=5).show()
Heatmap(df, row_col="country", col_col="theme").plot().show()
top_n_distribution(df, col="platform").show()
```

## Languages supported

| Language | Preprocessing | Sentence splitting | Translation |
|---|---|---|---|
| Arabic | ✓ (+ AraBERT) | ✓ | ✓ mBART / Helsinki |
| English | ✓ | ✓ | ✓ |
| Urdu | ✓ | ✓ | ✓ |
| Turkish | ✓ | ✓ | ✓ |
| Indonesian | ✓ | ✓ | ✓ |
| Hindi | ✓ | ✓ | ✓ |
| Persian/Farsi | ✓ | — | ✓ |
