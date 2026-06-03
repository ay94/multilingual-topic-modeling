import nltk
import time
import json
import math
import torch
import numpy as np
import pandas as pd
from umap import UMAP
from pathlib import Path
import plotly.express as px
from hdbscan import HDBSCAN
from bertopic import BERTopic
from tabulate import tabulate
from typing import Tuple, List
from tqdm.notebook import tqdm
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer



class SentenceRepresentation:
    """
    Base class for sentence representation extraction.
    """

    def __init__(self, model_name: str, data, text_col: str, file_name: str = 'embeddings'):
        self.model = self.load_model(model_name)
        self.sentences = self.extract_sentences(data, text_col)
        self.file_name = file_name
        self.embeddings: np.ndarray = np.array([])  # Initialize as empty array

    def load_model(self, model_name: str):
        """
        Load the model. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from data. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def extract_embeddings(self, batch_size):
        """
        Extract embeddings for the sentences. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def save_embeddings(self, directory: str) -> None:
        """
        Save the sentence embeddings to a JSON file.

        Parameters:
        - directory (str): The directory where the embeddings file will be saved.
        """
        file_path = Path(directory) / f'{self.file_name}.json'
        embeddings_list = [embedding.tolist() for embedding in self.embeddings]
        data = {"sentences": self.sentences, "embeddings": embeddings_list}

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
            print(f"Embeddings saved to {file_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, directory: str) -> Tuple[list, np.ndarray]:
        """
        Load the sentence embeddings from a JSON file.

        Parameters:
        - directory (str): The directory from where the embeddings file will be loaded.

        Returns:
        - Tuple containing a list of sentences and their corresponding embeddings as a numpy array.
        """
        file_path = Path(directory) / f'{self.file_name}.json'
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            embeddings = np.array(data['embeddings'])
            sentences = data['sentences']
            print(f"Embeddings loaded from {file_path}")
            return sentences, embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return [], np.array([])  # Return empty structures in case of error


class NomicRepresentation(SentenceRepresentation):
    """
    Representation class for extracting sentence embeddings using Sentence Transformers.
    """

    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a Sentence Transformer model.
        """
        return SentenceTransformer(model_name, trust_remote_code=True)

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from a pandas DataFrame column.
        """
        return data[text_col].apply(lambda x: f'clustering: {x}').tolist()

    def extract_embeddings(self, batch_size=100):
        """
        Extract embeddings using the loaded Sentence Transformer model.
        """
        extracted_embeddings = []
        num_batches = math.ceil(len(self.sentences) / batch_size)
        for i in tqdm(range(num_batches)):
            # Define the start and end of the current batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(self.sentences))

            # Extract the current batch
            batch_sentences = self.sentences[start_idx:end_idx]
            batch_embeddings = self.model.encode(batch_sentences, show_progress_bar=False)
            extracted_embeddings.append(batch_embeddings)
            torch.cuda.empty_cache()  # Clear unused memory

        self.embeddings = np.vstack(extracted_embeddings)


class GeneralRepresentation(SentenceRepresentation):
    """
    Representation class for extracting sentence embeddings using Sentence Transformers.
    """

    def load_model(self, model_name: str) -> SentenceTransformer:
        """
        Load a Sentence Transformer model.
        """
        return SentenceTransformer(model_name)

    def extract_sentences(self, data, text_col: str) -> List[str]:
        """
        Extract sentences from a pandas DataFrame column.
        """
        return data[text_col].tolist()

    def extract_embeddings(self):
        """
        Extract embeddings using the loaded Sentence Transformer model.
        """
        self.embeddings = self.model.encode(self.sentences, show_progress_bar=True)


class ParameterTuner:
    def __init__(self, embeddings, save_directory):
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty")

        self.embeddings = embeddings
        self.save_directory = save_directory

        self.umap_model = UMAP(random_state=1, metric='cosine', verbose=True)
        self.hdbscan_model = HDBSCAN(prediction_data=True)
        self.embedding_coordinates = self._create_coordinates()
        # Initialize additional attributes for parameter tuning
        self.n_neighbors = None
        self.n_components = None
        self.min_dist = None
        self.min_cluster_size = None
        self.cluster_selection_epsilon = None
        self.umap_embeddings = None
        self.hdbscan_labels = None

    def _create_coordinates(self):
        start_time = time.time()
        coordinates = self.umap_model.fit_transform(self.embeddings)
        print(f"--- {time.time() - start_time} seconds ---")
        return coordinates

    def tune_umap(self, n_neighbors=None, n_components=None, min_dist=None):
        if n_neighbors is not None:
            self.umap_model.n_neighbors = n_neighbors
            self.n_neighbors = n_neighbors

        if n_components is not None:
            self.umap_model.n_components = n_components
            self.n_components = n_components

        if min_dist is not None:
            self.umap_model.min_dist = min_dist
            self.min_dist = min_dist

    def tune_hdbscan(self, min_cluster_size=None, cluster_selection_epsilon=None):
        if min_cluster_size is not None:
            self.hdbscan_model.min_cluster_size = min_cluster_size
            self.min_cluster_size = min_cluster_size

        if cluster_selection_epsilon is not None:
            self.hdbscan_model.cluster_selection_epsilon = cluster_selection_epsilon
            self.cluster_selection_epsilon = cluster_selection_epsilon

    def apply_parameters(self):
        start_time = time.time()
        self.umap_embeddings = self.umap_model.fit_transform(self.embeddings)
        self.hdbscan_labels = self.hdbscan_model.fit_predict(self.umap_embeddings)
        start_time = time.time()
        print("--- %s seconds ---" % (time.time() - start_time))

    def visualize_parameters(self):
        hyperparameter_data = pd.DataFrame(self.embedding_coordinates, columns=['x', 'y'])
        hyperparameter_data['clusters'] = [f'cluster-({label})' for label in self.hdbscan_labels]
        print(f'Number of Clusters: {len(set(self.hdbscan_labels))}')
        topics_df = pd.DataFrame(Counter(self.hdbscan_labels).items(), columns=['topic', 'count'])
        topics_df["percentage"] = round(topics_df["count"] / topics_df["count"].sum() * 100, 2)
        topics_df.sort_values("percentage", ascending=False)
        print()
        print(tabulate(topics_df, headers='keys', tablefmt='grid'))
        print()
        print('Computing Silhouette Score')
        silhouette_avg = silhouette_score(self.umap_embeddings, self.hdbscan_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print()
        fig = px.scatter(hyperparameter_data, x="x", y="y", color="clusters")
        fig.show()

    def _ensure_directory_exists(self):
        path = Path(self.save_directory)
        if not path.exists():
            print(f"{path} does not exist. Creating...")
            path.mkdir(parents=True, exist_ok=True)

    def save_parameters(self, file_name):
        self._ensure_directory_exists()
        file_path = Path(self.save_directory) / f'{file_name}.json'
        data = {
            'embedding_coordinates': self.embedding_coordinates.tolist(),
            'n_neighbors': self.n_neighbors,
            'n_components': self.n_components,
            'min_cluster_size': self.min_cluster_size,
            'umap_embeddings': self.umap_embeddings.tolist(),
            'hdbscan_labels': self.hdbscan_labels.tolist(),
        }

        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
        except Exception as e:
            print(f"Error saving parameters: {e}")

    def load_parameters(self, file_name):
        file_path = Path(self.save_directory) / f'{file_name}.json'
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            self.embedding_coordinates = np.array(data['embedding_coordinates'])
            self.umap_model.n_neighbors = data['n_neighbors']
            self.umap_model.n_components = data['n_components']
            self.hdbscan_model.min_cluster_size = data['min_cluster_size']
            self.umap_embeddings = np.array(data['umap_embeddings'])
            self.hdbscan_labels = np.array(data['hdbscan_labels'])
        except Exception as e:
            print(f"Error loading parameters: {e}")


# Continue with other methods, applying similar principles.

class TopicModel():
    def __init__(self, embeddings, sentences, data, save_directory, paramaters):
        self.embeddings = embeddings
        self.sentences = sentences
        self.data = data
        self.save_directory = Path(save_directory)
        self.n_neighbors = paramaters.n_neighbors
        self.n_components = paramaters.n_components
        self.min_dist = paramaters.min_dist
        self.min_cluster_size = paramaters.min_cluster_size
        self.x = paramaters.embedding_coordinates[:,0]
        self.y = paramaters.embedding_coordinates[:,1]
        self.topic_model = None
        self.topics = None
        self.topic_info = pd.DataFrame()
        self.ensure_directory_exists(self.save_directory)

    def create_and_train_topic_model(self, lang='en'):
        start_time = time.time()
        umap_model = UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, random_state=1,
                          min_dist=self.min_dist, metric='cosine', verbose=True)
        hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size, prediction_data=True)
        if lang != 'en':
            self.topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True)
        else:
            nltk.download('stopwords')
            stop_words = stopwords.words(lang)
            vectorizer_model = CountVectorizer(stop_words=stop_words)
            self.topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                                        vectorizer_model=vectorizer_model, verbose=True)
        self.topics, _ = self.topic_model.fit_transform(self.sentences, embeddings=self.embeddings)
        print(f"Number of topics: {len(self.topics)}, Took --- %s seconds ---" % (time.time() - start_time))

    def save_all(self, topic_prefix):
        self.save_annotated_data(topic_prefix)
        self.save_topic_info()
        self.save_topic_model()

    def save_topic_info(self):
        self.topic_info = self.topic_model.get_topic_info()
        topic_info_path = self.save_directory / 'topic_info.csv'
        self.topic_info.to_csv(topic_info_path, index=False)
        print(f"Saved topic info to {topic_info_path}")

    def save_annotated_data(self, topic_prefix):
        annotated_data_path = self.save_directory / 'annotated_data.json'
        self.data[f'{topic_prefix}_x'] = self.x
        self.data[f'{topic_prefix}_y'] = self.y
        self.data[f'{topic_prefix}_topic'] = self.topics
        self.data.to_json(annotated_data_path, lines=True, orient='records')
        print(f"Saved annotated data to {annotated_data_path}")

    def check_model_output(self, embedding_extractor, paramater_tuner, plot=False):
        start_time = time.time()
        umap_embeddings = self.topic_model.umap_model.fit_transform(embedding_extractor.embeddings)
        hdbscan_labels = self.topic_model.hdbscan_model.fit_predict(umap_embeddings)
        start_time = time.time()
        print("--- %s seconds ---" % (time.time() - start_time))
        print()
        print(f'Number of Clusters: {len(set(self.topics))}')
        print()
        topics_df = pd.DataFrame(Counter(self.topics).items(), columns=['topic', 'count'])
        topics_df["percentage"] = round(topics_df["count"] / topics_df["count"].sum() * 100, 2)
        topics_df.sort_values("percentage", ascending=False)
        print()
        hyperparameter_data = pd.DataFrame(paramater_tuner.embedding_coordinates, columns=['x', 'y'])
        hyperparameter_data['clusters'] = [f'cluster-({label})' for label in hdbscan_labels]
        if plot:
            fig = px.scatter(hyperparameter_data, x="x", y="y", color="clusters")
            fig.show()
            return fig

    def save_topic_model(self):
        model_path = self.save_directory / f'topic_model-{len(set(self.topics))}.bertopic'
        self.topic_model.save(model_path)
        print(f"Saved BERTopic model to {model_path}")

    def load_topic_model(self, model_path):
        model_path = self.save_directory / model_path
        self.topic_model = BERTopic.load(model_path)
        print(f"Loaded BERTopic model from {model_path}")

    def generate_samples(self, topic_prefix, frac=0.1):
        sample_data_path = self.save_directory / f'annotate_sample_data-({frac * 100})%.xlsx'
        topic_info = self.topic_info.rename(columns={'Topic': f'{topic_prefix}_topic'})
        sampling_data = self.data.merge(topic_info, on=f'{topic_prefix}_topic')
        samples = []
        for tp, tp_df in sampling_data.groupby(f'{topic_prefix}_topic'):
            if len(tp_df) * frac < 20:
                samples.append(tp_df)  # If 10% of the sample is less than 20, take the whole dataset
            else:
                samples.append(tp_df.sample(frac=frac, random_state=1))  # Otherwise, take a sample as per the fraction
        sample_data = pd.concat(samples)
        sample_data.to_excel(sample_data_path, index=False)

    @staticmethod
    def ensure_directory_exists(path):
        path.mkdir(parents=True, exist_ok=True)


class SentenceRepresentation_deprecated:
    def __init__(self, model_name, data, text_col, file_name='embeddings'):
        self.model = SentenceTransformer(model_name)
        self.sentences = data[text_col].tolist()
        self.file_name = file_name

    def extract_embeddings(self):
        self.embeddings = self.model.encode(self.sentences, show_progress_bar=True)

    def save_embeddings(self, directory):
        file_path = Path(directory) / f'{self.file_name}.json'
        embeddings_list = [embedding.tolist() for embedding in self.embeddings]
        data = {"sentences": self.sentences, "embeddings": embeddings_list}

        try:
            if not file_path.exists():
                print(f"{file_path} does not exist. Creating...")
                file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, directory):
        file_path = Path(directory) / f'{self.file_name}.json'
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            embeddings = np.array(data['embeddings'])
            sentences = data['sentences']
            return sentences, embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return [], np.array([])
