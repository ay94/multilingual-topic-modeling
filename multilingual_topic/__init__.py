"""
multilingual-topic-toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~
A toolkit for multilingual topic modelling, machine translation,
SetFit-guided cluster refinement, and text preprocessing.

Originally developed for large-scale social media and news analysis across
Arabic, English, Urdu, Farsi, Turkish, Hindi and Indonesian corpora.
"""

from .topic_modeling import SentenceRepresentation, TopicModel, ParameterTuner
from .translation import ManyToManyTranslator, HelsinkiTranslator, translate_iterator
from .classification import (
    train_setfit,
    cross_validate_setfit,
    Evaluation,
    ClassifierValidationHelper,
    ClassifierAnnotationHelper,
    ClassifierTestingHelper,
    DataVisualizer,
)
from .preprocessing import TextPreprocessor
from .checks import TruncationChecks, SentenceSplitterGenerator, TextChunker
from .utils import FileHandler
from .analysis import volume_over_time, top_n_distribution, platform_language_breakdown, Heatmap, top_accounts_by_country

__version__ = "0.1.0"
__all__ = [
    "SentenceRepresentation",
    "TopicModel",
    "ParameterTuner",
    "ManyToManyTranslator",
    "HelsinkiTranslator",
    "translate_iterator",
    "ClassifierValidationHelper",
    "ClassifierAnnotationHelper",
    "ClassifierTestingHelper",
    "DataVisualizer",
    "TextPreprocessor",
    "TruncationChecks",
    "SentenceSplitterGenerator",
    "TextChunker",
    "FileHandler",
]
