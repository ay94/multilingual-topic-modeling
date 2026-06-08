import numpy as np
import pandas as pd
from umap import UMAP
from typing import List, Optional, Dict
import plotly.express as px
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


# ── Training ──────────────────────────────────────────────────────────────────

def train_setfit(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    num_epochs: int = 1,
    batch_size: int = 16,
    seed: int = 1,
) -> SetFitModel:
    """
    Fine-tune a SetFit model for few-shot cluster refinement.

    Used between BERTopic layers to steer the embedding space toward
    analytically useful cluster boundaries. Provide positive/negative
    examples (relevant/irrelevant) drawn from cluster output.

    Tested backbones:
        - sentence-transformers/all-mpnet-base-v2
        - nomic-ai/nomic-embed-text-v1
        - WhereIsAI/UAE-Large-V1
        - sentence-transformers/paraphrase-mpnet-base-v2

    Args:
        train_dataset: HF Dataset with 'text' and 'label' columns.
        eval_dataset: Optional validation dataset.
        model_name: Sentence transformer backbone.
        num_epochs: Training epochs (1 is typically sufficient for few-shot).
        batch_size: Contrastive training batch size.
        seed: Random seed for reproducibility.

    Returns:
        Fine-tuned SetFitModel.
    """
    args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="no",
        seed=seed,
    )
    model = SetFitModel.from_pretrained(model_name)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
    )
    trainer.train()
    return model


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_setfit(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    n_folds: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    num_epochs: int = 1,
) -> Dict[str, List[float]]:
    """
    K-fold cross-validation for a SetFit binary classifier.

    Each fold shuffles the data with a different seed, trains a fresh model,
    and evaluates on a held-out test split.

    Args:
        df: DataFrame with text and label columns.
        text_col: Column containing the text.
        label_col: Column containing labels (e.g. 'relevant'/'irrelevant').
        model_name: Sentence transformer backbone.
        n_folds: Number of folds.
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction used for validation (remainder = test).
        batch_size: Training batch size.
        num_epochs: Training epochs per fold.

    Returns:
        Dict with lists of accuracy, precision, recall and F1 across folds.
    """
    results: Dict[str, List[float]] = {
        "accuracy": [], "precision": [], "recall": [], "f1": []
    }

    for i in range(n_folds):
        shuffled = df.sample(frac=1, random_state=i).reset_index(drop=True)
        n = len(shuffled)
        train_end = round(train_ratio * n)
        val_end = round((train_ratio + val_ratio) * n)

        train_ds = Dataset.from_pandas(shuffled[:train_end][[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}))
        val_ds   = Dataset.from_pandas(shuffled[train_end:val_end][[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}))
        test_ds  = Dataset.from_pandas(shuffled[val_end:][[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}))

        model = train_setfit(train_ds, val_ds, model_name=model_name, batch_size=batch_size, num_epochs=num_epochs, seed=i)

        y_true = test_ds["label"]
        y_pred = model(test_ds["text"])

        results["accuracy"].append(accuracy_score(y_true, y_pred))
        results["precision"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        results["recall"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
        results["f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))

        print(f"Fold {i+1}/{n_folds} — F1: {results['f1'][-1]:.4f}")

    for metric, scores in results.items():
        print(f"{metric.capitalize():12s}: mean={np.mean(scores):.4f}  min={np.min(scores):.4f}  max={np.max(scores):.4f}")

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

class Evaluation:
    """
    Evaluation helper for a trained SetFit classifier.

    Usage::

        ev = Evaluation(y_true, y_pred)
        ev.report()
        fig = ev.confusion_matrix()
        fig.show()
    """

    def __init__(self, y_true: List, y_pred: List):
        self.y_true = list(y_true)
        self.y_pred = list(y_pred)

    def report(self) -> None:
        print(classification_report(self.y_true, self.y_pred))

    def confusion_matrix(self, fig_size: tuple = (600, 600)) -> px.imshow:
        labels = sorted(set(self.y_true))
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        fig = px.imshow(
            cm_df,
            color_continuous_scale="rdbu_r",
            title="Confusion Matrix",
        )
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="True")
        fig.update_layout(width=fig_size[0], height=fig_size[1])
        return fig

    def metrics(self) -> Dict[str, float]:
        return {
            "accuracy":  accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, average="macro", zero_division=0),
            "recall":    recall_score(self.y_true, self.y_pred, average="macro", zero_division=0),
            "f1":        f1_score(self.y_true, self.y_pred, average="macro", zero_division=0),
        }


class ClassifierBaseHelper(ABC):
    def __init__(self, data: pd.DataFrame, src_text: str, text_col: str, label_col: str, prediction_col: str,
                 topic_col: str):
        self.data = data
        self.src_text = src_text
        self.text_col = text_col
        self.label_col = label_col
        self.prediction_col = prediction_col
        self.topic_col = topic_col

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def split_data(self, cleaned_data):
        pass

    @abstractmethod
    def convert_split_to_dataset(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, sample_size: int):
        pass

    def _visualize_2d(self, embeddings_2d: np.array, df: pd.DataFrame, hover_cols: List[str], color_col: str,
                      size_col: str, title: str):
        """
    Visualize embeddings in 2D with plotly
    :param df: pd.DataFrame containing meta-data (data relating to hover, color, size, etc.)
        The order of the DataFrame must match the order of the embeddings.
    :param embeddings_2d: 2D embeddings, shape (n, 2)
    :param hover_cols: Columns to display in the hover tooltip
    :param color_col: Column to use for coloring the points
    :param size_col: Column to use for sizing the points
    :param title: Title of the plot
    :return: px.scatter plot
    """
        assert len(df) == embeddings_2d.shape[0]
        assert embeddings_2d.shape[1] == 2

        vis_cols = []
        if color_col:
            vis_cols.append(color_col)
        if size_col:
            vis_cols.append(size_col)
        if hover_cols:
            vis_cols.extend(hover_cols)

        df_vis = df[vis_cols].copy()
        df_vis['x'] = embeddings_2d[:, 0]
        df_vis['y'] = embeddings_2d[:, 1]
        fig = px.scatter(df_vis, x='x', y='y',
                         color=color_col, size=size_col,
                         hover_data=hover_cols, title=title)
        return fig

    def visualize_data(self, model, dataset):
        umap = UMAP(
            random_state=1,
            metric='cosine',
        )

        embeddings = model.model_body.encode(dataset[self.text_col], show_progress_bar=True)
        embeddings_umap = umap.fit_transform(embeddings)
        return self._visualize_2d(embeddings_umap, dataset.to_pandas(), hover_cols=[self.text_col],
                                  color_col=self.label_col, size_col=None, title='')


class ClassifierValidationHelper(ClassifierBaseHelper):
    def __init__(self, data: pd.DataFrame, theme_col: str, theme: str, src_text: str, text_col: str,
                 label_col: str, prediction_col: str, topic_col: str, sample_size: int = 10):
        super().__init__(data, src_text, text_col, label_col, prediction_col, topic_col)
        self.theme_col = theme_col
        self.theme = theme
        self.sample_size = sample_size

    def prepare_data(self, ):
        cleaned_data = self.data[self.data[self.theme_col] != ''].copy()

        conditions = [
            cleaned_data[self.theme_col] == self.theme,
            cleaned_data[self.theme_col] != self.theme,
        ]

        labels = ['relevant', 'irrelevant']

        cleaned_data[self.text_col] = cleaned_data[self.src_text].copy()

        cleaned_data[self.label_col] = np.select(conditions, labels, default='Unknown')

        self.topics = cleaned_data[self.topic_col]
        print(self.topics)

        return cleaned_data[[self.text_col, self.label_col, self.topic_col]]

    def split_data(self, cleaned_data):
        # Splitting the data into training and validation sets
        train_data, valid_data, train_labels, valid_labels = train_test_split(
            cleaned_data[self.text_col],  # Features (text data)
            cleaned_data[self.label_col],  # Target variable (labels)
            test_size=0.2,  # Specifies the proportion of data used for validation
            random_state=42,  # For reproducibility of results
            stratify=cleaned_data[self.label_col]  # To maintain the same distribution of labels in both sets
        )

        # If you need to combine the features and labels back into DataFrame structures
        train_df = pd.DataFrame({self.text_col: train_data, self.label_col: train_labels})
        valid_df = pd.DataFrame({self.text_col: valid_data, self.label_col: valid_labels})
        return train_df, valid_df

    def convert_split_to_dataset(self, train_df: pd.DataFrame(), valid_df: pd.DataFrame(), sample_size: int):
        vl_dataset = Dataset.from_pandas(train_df)
        tr_dataset = Dataset.from_pandas(valid_df)
        train_dataset = sample_dataset(vl_dataset, label_column=self.label_col, num_samples=sample_size)
        valid_dataset = sample_dataset(tr_dataset, label_column=self.label_col, num_samples=sample_size)
        return train_dataset, valid_dataset

    def process_and_validate(self, model):
        # Step 1: Prepare the data
        cleaned_data = self.prepare_data()

        # Step 2: Split the data into training and validation sets
        train_df, valid_df = self.split_data(cleaned_data)

        # Step 3: Convert the split data into datasets suitable for training
        self.train_dataset, self.valid_dataset = self.convert_split_to_dataset(train_df, valid_df, self.sample_size)

        # Optionally, Step 4: Train the model using the datasets (This part depends on your model training setup)
        # model.train(train_dataset)

        # Step 5: Return or visualize results
        # return some_results_or_model
        return self.visualize_data(model, self.valid_dataset)  # Example of visualization

    def annotate_data(self, model):
        # Step 1: Prepare the data
        predictions = model(self.valid_dataset[self.text_col])
        df = self.valid_dataset.to_pandas()
        df[self.prediction_col] = predictions
        print(self.topic_col, self.topics)
        df[self.topic_col] = self.topics

        # Step 5: Return or visualize results
        # return some_results_or_model
        return df[[self.topic_col, self.text_col, self.label_col, self.prediction_col]]

class ClassifierAnnotationHelper(ClassifierBaseHelper):
    def __init__(self, data: pd.DataFrame, theme_col: str, theme: str, src_text: str, text_col: str,
                 label_col: str, prediction_col: str, topic_col: str, sample_size: int = 10):
        super().__init__(data, src_text, text_col, label_col, prediction_col, topic_col)
        self.theme_col = theme_col
        self.theme = theme
        self.sample_size = sample_size

    def prepare_data(self, ):
        cleaned_data = self.data[self.data[self.theme_col] != ''].copy()

        conditions = [
            cleaned_data[self.theme_col] == self.theme,
            cleaned_data[self.theme_col] != self.theme,
        ]

        labels = ['relevant', 'irrelevant']

        cleaned_data[self.text_col] = cleaned_data[self.src_text].copy()

        cleaned_data[self.label_col] = np.select(conditions, labels, default='Unknown')

        self.topics = cleaned_data[self.topic_col]

        return cleaned_data[[self.text_col, self.label_col, self.topic_col]]

    def split_data(self, cleaned_data):
        # Splitting the data into training and validation sets
        train_data, valid_data, train_labels, valid_labels, train_topic, valid_topic = train_test_split(
            cleaned_data[self.text_col],  # Features (text data)
            cleaned_data[self.label_col],  # Target variable (labels)
            cleaned_data[self.topic_col],  # Target variable (labels)
            test_size=0.2,  # Specifies the proportion of data used for validation
            random_state=42,  # For reproducibility of results
            stratify=cleaned_data[self.label_col]  # To maintain the same distribution of labels in both sets
        )

        # If you need to combine the features and labels back into DataFrame structures
        train_df = pd.DataFrame({self.text_col: train_data, self.label_col: train_labels, self.topic_col: train_topic})
        valid_df = pd.DataFrame({self.text_col: valid_data, self.label_col: valid_labels, self.topic_col: valid_topic})
        return train_df, valid_df

    def convert_split_to_dataset(self, train_df: pd.DataFrame(), valid_df: pd.DataFrame(), sample_size: int):
        vl_dataset = Dataset.from_pandas(train_df)
        tr_dataset = Dataset.from_pandas(valid_df)
        train_dataset = sample_dataset(vl_dataset, label_column=self.label_col, num_samples=sample_size)
        valid_dataset = sample_dataset(tr_dataset, label_column=self.label_col, num_samples=sample_size)
        return train_dataset, valid_dataset

    def process_and_validate(self, model):
        # Step 1: Prepare the data
        cleaned_data = self.prepare_data()

        # Step 2: Split the data into training and validation sets
        train_df, valid_df = self.split_data(cleaned_data)

        # Step 3: Convert the split data into datasets suitable for training
        self.train_dataset, self.valid_dataset = self.convert_split_to_dataset(train_df, valid_df, self.sample_size)

        # Optionally, Step 4: Train the model using the datasets (This part depends on your model training setup)
        # model.train(train_dataset)

        # Step 5: Return or visualize results
        # return some_results_or_model
        return self.visualize_data(model, self.valid_dataset)  # Example of visualization

    def annotate_data(self, model):
        # Step 1: Prepare the data
        predictions = model(self.valid_dataset[self.text_col])
        df = self.valid_dataset.to_pandas()
        df[self.prediction_col] = predictions

        # Step 5: Return or visualize results
        # return some_results_or_model
        return df[[self.topic_col, self.text_col, self.label_col, self.prediction_col]]


class ClassifierTestingHelper(ClassifierBaseHelper):
    def __init__(self, data: pd.DataFrame, src_text: str, text_col: str,
                 label_col: str, prediction_col: str, topic_col: str, ):
        super().__init__(data, src_text, text_col, label_col, prediction_col, topic_col)

    def prepare_data(self, ):
        cleaned_data = self.data[self.data[self.src_text] != ''].copy()

        cleaned_data[self.text_col] = cleaned_data[self.src_text].copy()

        cleaned_data[self.label_col] = 'Unknown'

        test_data = cleaned_data[[self.text_col, self.label_col, self.topic_col]]


        return Dataset.from_pandas(test_data)

    def split_data(self, cleaned_data):
        raise NotImplementedError("Splitting data is not applicable for testing scenarios.")

    def convert_split_to_dataset(self, train_df: pd.DataFrame(), valid_df: pd.DataFrame(), sample_size: int = 10):
        raise NotImplementedError("Converting split data to datasets is not applicable for testing scenarios.")

    def process_and_validate(self, model):
        # Step 1: Prepare the data
        self.test_dataset = self.prepare_data()

        # Step 5: Return or visualize results
        # return some_results_or_model
        return self.visualize_data(model, self.test_dataset)  # Example of visualization

    def annotate_data(self, model):
        # Step 1: Prepare the data
        predictions = model(self.test_dataset[self.text_col])
        df = self.test_dataset.to_pandas()
        df[self.prediction_col] = predictions

        # Step 5: Return or visualize results
        # return some_results_or_model
        return df[[self.topic_col, self.text_col, self.label_col, self.prediction_col]]


class DataVisualizer():
  def __init__(self, text_col, label_col):
    self.text_col = text_col
    self.label_col = label_col


  def _visualize_2d(self, embeddings_2d: np.array, df: pd.DataFrame, hover_cols: List[str],
                    size_col: str, title: str, color_col: str='label',):
      """
      Visualize embeddings in 2D with plotly
      :param df: pd.DataFrame containing meta-data (data relating to hover, color, size, etc.)
          The order of the DataFrame must match the order of the embeddings.
      :param embeddings_2d: 2D embeddings, shape (n, 2)
      :param hover_cols: Columns to display in the hover tooltip
      :param color_col: Column to use for coloring the points
      :param size_col: Column to use for sizing the points
      :param title: Title of the plot
      :return: px.scatter plot
      """
      assert len(df) == embeddings_2d.shape[0]
      assert embeddings_2d.shape[1] == 2

      vis_cols = []
      if color_col:
          vis_cols.append(color_col)
      if size_col:
          vis_cols.append(size_col)
      if hover_cols:
          vis_cols.extend(hover_cols)

      df_vis = df[vis_cols].copy()
      df_vis['x'] = embeddings_2d[:, 0]
      df_vis['y'] = embeddings_2d[:, 1]
      fig = px.scatter(df_vis, x='x', y='y',
                        color=color_col, size=size_col,
                        hover_data=hover_cols, title=title)
      return fig
  def visualize_data(self, model, dataset):
      umap = UMAP(
          random_state=1,
          metric='cosine',
      )

      embeddings = model.model_body.encode(dataset[self.text_col], show_progress_bar=True)
      embeddings_umap = umap.fit_transform(embeddings)
      return self._visualize_2d(embeddings_umap, dataset.to_pandas(), hover_cols=[self.text_col],
                                color_col=self.label_col, size_col=None, title='')
