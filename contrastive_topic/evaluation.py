"""
evaluation.py
-------------
Classifier evaluation utilities — no heavy dependencies required.
Works with any sklearn-compatible y_true / y_pred lists.
"""

from typing import List, Dict
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


class Evaluation:
    """
    Evaluation helper for binary or multi-class classifiers.

    Usage::

        ev = Evaluation(y_true, y_pred)
        ev.report()
        fig = ev.confusion_matrix()
        fig.show()
        metrics = ev.metrics()
    """

    def __init__(self, y_true: List, y_pred: List):
        self.y_true = list(y_true)
        self.y_pred = list(y_pred)

    def report(self) -> None:
        print(classification_report(self.y_true, self.y_pred))

    def confusion_matrix(self, fig_size: tuple = (600, 600)):
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
