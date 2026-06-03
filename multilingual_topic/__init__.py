"""
multilingual-topic-toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~
A toolkit for multilingual topic modelling, machine translation,
SetFit-guided classification, and text preprocessing.

Originally developed for large-scale social media analysis across
Arabic, English, Urdu, Farsi, Turkish and Indonesian corpora.
"""

from .topic_modeling import SentenceRepresentation, BERTopicModel
from .translation import ManyToManyTranslator, HelsinkiTranslator, translate_iterator
from .classification import ClassifierValidationHelper, ClassifierTestingHelper, DataVisualizer
from .preprocessing import TextPreprocessor
from .checks import TruncationChecks, SentenceSplitterGenerator, TextChunker

__version__ = "0.1.0"
__all__ = [
    "SentenceRepresentation",
    "BERTopicModel",
    "ManyToManyTranslator",
    "HelsinkiTranslator",
    "translate_iterator",
    "ClassifierValidationHelper",
    "ClassifierTestingHelper",
    "DataVisualizer",
    "TextPreprocessor",
    "TruncationChecks",
    "SentenceSplitterGenerator",
    "TextChunker",
]
