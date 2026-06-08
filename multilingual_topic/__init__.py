"""
multilingual-topic-modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multilingual topic modelling with BERTopic, outlier mitigation,
SetFit-guided cluster refinement, machine translation, and text preprocessing.
"""

__version__ = "0.1.0"

# Light modules — always available
from .preprocessing import TextPreprocessor
from .evaluation import Evaluation
from .checks import TruncationChecks, SentenceSplitterGenerator, TextChunker
from .utils import FileHandler
from .analysis import (
    volume_over_time,
    top_n_distribution,
    platform_language_breakdown,
    Heatmap,
    top_accounts_by_country,
)
from .translation import ManyToManyTranslator, HelsinkiTranslator, translate_iterator

# Heavy modules — require bertopic, umap-learn, hdbscan
try:
    from .topic_modeling import SentenceRepresentation, TopicModel, ParameterTuner
    from .outlier_mitigation import StaticReducer, StaticClusterer, SoftReclusterer
except ImportError:
    pass

# SetFit modules — require setfit, sentence-transformers
try:
    from .setfit import (
        train_setfit,
        cross_validate_setfit,
        Evaluation,
        ClassifierValidationHelper,
        ClassifierAnnotationHelper,
        ClassifierTestingHelper,
        DataVisualizer,
    )
except ImportError:
    # Evaluation only needs sklearn — import it separately
    try:
        from .setfit import Evaluation
    except ImportError:
        pass
