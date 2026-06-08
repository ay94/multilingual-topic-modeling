import numpy as np
import networkx as nx
from collections import Counter


class StaticReducer:
    """
    Drop-in replacement for a UMAP model that returns pre-computed embeddings.

    Pass this to BERTopic as `umap_model` when you already have UMAP coordinates
    and want to skip re-running dimensionality reduction.
    """

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def fit_transform(self, X):
        assert X.shape[0] == len(self.embeddings)
        return self.embeddings

    def transform(self, X):
        assert X.shape[0] == len(self.embeddings)
        return self.embeddings

    def fit(self, X):
        assert X.shape[0] == len(self.embeddings)
        return self

    def __repr__(self):
        return "StaticReducer()"


class StaticClusterer:
    """
    Drop-in replacement for an HDBSCAN model that returns pre-computed labels.

    Pass this to BERTopic as `hdbscan_model` when you already have cluster
    assignments and want to skip re-running clustering.
    """

    def __init__(self, labels: np.ndarray):
        self.labels_ = labels

    def fit_predict(self, X):
        assert X.shape[0] == len(self.labels_)
        return self.labels_

    def fit(self, X):
        assert X.shape[0] == len(self.labels_)
        return self

    def __repr__(self):
        return "StaticClusterer()"


class SoftReclusterer:
    """
    Reassign HDBSCAN -1 outliers to fringe clusters using membership vectors.

    HDBSCAN assigns a membership vector to every point — a score per cluster
    indicating how strongly that point is associated with each dense region.
    Points labelled -1 still have non-zero membership scores; this class uses
    those scores to assign them to one or more 'fringe' clusters rather than
    discarding them as noise.

    Parameters
    ----------
    original_labels : array-like
        HDBSCAN topic labels for each document (-1 for outliers).
    membership_vectors : np.ndarray, shape (n_docs, n_topics)
        HDBSCAN membership/probability matrix (from BERTopic's `calculate_probabilities=True`).
    method : str
        'fixed' — use `min_membership_score` as a hard threshold.
        'ratio' — use `max_score * threshold_ratio` as the threshold per point.
    threshold_ratio : float
        Used when method='ratio'. A cluster qualifies as fringe if its score
        is >= max_score * threshold_ratio. Prevents assignments when scores
        are closely bunched (high variability guard).
    min_membership_score : float
        Minimum score a cluster must have to qualify. Points below this on all
        clusters remain outliers.
    max_core_clusters : int
        Maximum number of core clusters a point can be assigned to.
    min_fringe_cluster_size : int
        Minimum number of points in a fringe group to keep it as a fringe cluster.
        Groups smaller than this are reclassified as outliers.
    """

    def __init__(
        self,
        original_labels,
        membership_vectors,
        method='fixed',
        threshold_ratio=0.5,
        min_membership_score=0.001,
        max_core_clusters=5,
        min_fringe_cluster_size=2,
    ):
        self.original_labels = np.array(original_labels)
        self.membership_vectors = membership_vectors
        self.method = method
        self.min_membership_score = min_membership_score
        self.max_core_clusters = max_core_clusters
        self.min_fringe_cluster_size = min_fringe_cluster_size
        self.threshold_ratio = threshold_ratio

        self.number_of_core_clusters = self.membership_vectors.shape[1]
        self.outlier_cluster_id = -1

        self.label_to_fringe_cluster_map = None   # new_label -> frozenset of core labels
        self.fringe_cluster_to_label_map = None   # tuple of cores -> new_label
        self.fringe_cluster_counts = None
        self.labels_ = None

    def find_fringe_clusters(self, membership_vector):
        if self.method == 'fixed':
            return self._find_fringe_clusters_fixed(membership_vector)
        elif self.method == 'ratio':
            return self._find_fringe_clusters_ratio(membership_vector)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'fixed' or 'ratio'.")

    def _find_fringe_clusters_fixed(self, membership_vector):
        if membership_vector.max() < self.min_membership_score:
            return False, self.outlier_cluster_id

        above = np.where(membership_vector > self.min_membership_score)[0]
        top = above[np.argsort(membership_vector[above])[-self.max_core_clusters:]]
        return True, tuple(sorted(top))

    def _find_fringe_clusters_ratio(self, membership_vector):
        """
        Threshold = max_score * threshold_ratio.

        Example: max score 0.6, ratio 0.5 → threshold 0.3.
        Any cluster with score >= 0.3 qualifies as fringe for this point.
        """
        if membership_vector.max() < self.min_membership_score:
            return False, self.outlier_cluster_id

        threshold = self.threshold_ratio * membership_vector.max()
        above = np.where(membership_vector > threshold)[0]
        top = above[np.argsort(membership_vector[above])[-self.max_core_clusters:]]
        return True, tuple(sorted(top))

    def fit(self, X):
        assert X.shape[0] == self.membership_vectors.shape[0]

        # Find candidate fringe cluster for each -1 point
        clusters = [
            self.find_fringe_clusters(m) if orig == -1 else (False, orig)
            for m, orig in zip(self.membership_vectors, self.original_labels)
        ]

        # Drop fringe groups that are too small
        all_counts = Counter(cs for is_fringe, cs in clusters if is_fringe)
        self.fringe_cluster_counts = {
            cs: cnt for cs, cnt in all_counts.items()
            if cnt >= self.min_fringe_cluster_size
        }

        # Points whose fringe group was too small become true outliers
        clusters = [
            (False, self.outlier_cluster_id) if (is_fringe and cs not in self.fringe_cluster_counts)
            else (is_fringe, cs)
            for is_fringe, cs in clusters
        ]

        # Assign new integer labels to each valid fringe group
        self.label_to_fringe_cluster_map = {
            self.number_of_core_clusters + i: cs
            for i, cs in enumerate(self.fringe_cluster_counts)
        }
        self.fringe_cluster_to_label_map = {cs: l for l, cs in self.label_to_fringe_cluster_map.items()}

        self.labels_ = np.array([
            self.fringe_cluster_to_label_map[cs] if is_fringe else cs
            for is_fringe, cs in clusters
        ])
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def build_display_name(self, cluster_id):
        """Return a human-readable label: 'cluster-N', 'fringe-A-B-C', or 'outlier'."""
        cluster_id = int(cluster_id)
        if cluster_id == self.outlier_cluster_id:
            return 'outlier'
        elif cluster_id in self.label_to_fringe_cluster_map:
            cores = '-'.join(str(c) for c in self.label_to_fringe_cluster_map[cluster_id])
            return f'fringe-{cores}'
        elif cluster_id < self.number_of_core_clusters:
            return f'cluster-{cluster_id}'
        else:
            raise ValueError(f'Unknown cluster ID {cluster_id}')

    def fringe_linkage_network(self):
        """
        Build a networkx graph connecting fringe clusters to their core clusters.

        Each fringe node has edges to every core cluster it straddles. Edge weight
        reflects how many points are in the fringe group.
        """
        G = nx.Graph()

        for label, count in Counter(self.labels_).items():
            G.add_node(label, size=count, label=self.build_display_name(label))

        for label, core_set in self.label_to_fringe_cluster_map.items():
            count = self.fringe_cluster_counts.get(core_set, 0)
            for core in core_set:
                G.add_edge(label, core, weight=count)

        return G

    def summary(self):
        """Print a breakdown of fringe clusters, core associations, and residual outliers."""
        if self.labels_ is None:
            print("Not fitted yet. Call fit() or fit_predict() first.")
            return

        total = len(self.labels_)
        n_outlier = int(np.sum(self.labels_ == self.outlier_cluster_id))
        n_core = int(np.sum(self.labels_ < self.number_of_core_clusters))
        n_fringe = total - n_outlier - n_core

        print(f"Total documents : {total}")
        print(f"Core assignments: {n_core} ({n_core/total*100:.1f}%)")
        print(f"Fringe clusters : {n_fringe} docs across {len(self.fringe_cluster_counts)} groups ({n_fringe/total*100:.1f}%)")
        print(f"Residual outliers: {n_outlier} ({n_outlier/total*100:.1f}%)")
        print()
        print("Fringe groups (core associations → size):")
        for cs, cnt in sorted(self.fringe_cluster_counts.items(), key=lambda x: -x[1]):
            label = self.fringe_cluster_to_label_map[cs]
            print(f"  {self.build_display_name(label):30s}  {cnt} docs")
