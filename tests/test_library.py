"""
Tests for contrastive-topic-modeling.
Covers all modules that don't require model downloads.
Model-dependent tests are marked with @pytest.mark.slow and skipped by default.
"""

import pytest
import numpy as np
import pandas as pd
from datasets import Dataset


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        from contrastive_topic import TextPreprocessor
        self.pp = TextPreprocessor()

    def test_removes_links(self):
        assert "http://example.com" not in self.pp.preprocess("check http://example.com out")

    def test_removes_mentions(self):
        assert "@user" not in self.pp.preprocess("hello @user how are you")

    def test_removes_hashtags(self):
        result = self.pp.preprocess("#Sweden is great")
        assert "#" not in result

    def test_removes_emojis(self):
        result = self.pp.preprocess("hello 😀 world")
        assert "😀" not in result

    def test_normalises_spaces(self):
        result = self.pp.preprocess("too  many   spaces")
        assert "  " not in result

    def test_empty_string(self):
        result = self.pp.preprocess("")
        assert result == ""

    def test_arabic_text_passthrough(self):
        text = "السويد تواجه انتقادات"
        result = self.pp.preprocess(text)
        assert len(result) > 0


# ── Sentence splitting ────────────────────────────────────────────────────────

class TestSentenceSplitter:
    def setup_method(self):
        from contrastive_topic import SentenceSplitterGenerator
        self.gen = SentenceSplitterGenerator

    def test_english_splits_on_period(self):
        splitter = self.gen.get_splitter("en")
        sentences = splitter.simple_split("Hello world. How are you?")
        assert len(sentences) >= 2

    def test_arabic_splitter_exists(self):
        splitter = self.gen.get_splitter("ar")
        assert splitter is not None

    def test_indonesian_splitter_exists(self):
        splitter = self.gen.get_splitter("id")
        assert splitter is not None

    def test_unsupported_language_raises(self):
        with pytest.raises(NotImplementedError):
            self.gen.get_splitter("xx")


# ── Analysis ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "date":     pd.date_range("2022-01-01", periods=n, freq="6h"),
        "country":  np.random.choice(["Egypt", "Turkey", "Pakistan", "Indonesia"], n),
        "theme":    np.random.choice(["Geopolitics", "Religion", "Welfare", "Media"], n),
        "platform": np.random.choice(["Twitter", "Facebook", "Telegram"], n),
        "account":  [f"account_{i % 20}" for i in range(n)],
    })


class TestVolumeOverTime:
    def test_returns_figure(self, dummy_df):
        from contrastive_topic import volume_over_time
        fig = volume_over_time(dummy_df, date_col="date")
        assert fig is not None

    def test_with_group_col(self, dummy_df):
        from contrastive_topic import volume_over_time
        fig = volume_over_time(dummy_df, date_col="date", group_col="country", top_n=3)
        assert fig is not None

    def test_handles_bad_dates(self, dummy_df):
        from contrastive_topic import volume_over_time
        dummy_df = dummy_df.copy()
        dummy_df.loc[0, "date"] = "not-a-date"
        fig = volume_over_time(dummy_df, date_col="date")
        assert fig is not None


class TestTopNDistribution:
    def test_returns_figure(self, dummy_df):
        from contrastive_topic import top_n_distribution
        fig = top_n_distribution(dummy_df, col="country", top_n=4)
        assert fig is not None


class TestHeatmap:
    def test_pivot_table_shape(self, dummy_df):
        from contrastive_topic import Heatmap
        hm = Heatmap(dummy_df, row_col="country", col_col="theme")
        pivot = hm.get_pivot_table()
        assert pivot.shape[0] <= 4  # 4 countries
        assert pivot.shape[1] <= 4  # 4 themes

    def test_plot_returns_figure(self, dummy_df):
        from contrastive_topic import Heatmap
        hm = Heatmap(dummy_df, row_col="country", col_col="theme")
        fig = hm.plot()
        assert fig is not None


class TestTopAccountsByCountry:
    def test_returns_dataframe(self, dummy_df):
        from contrastive_topic import top_accounts_by_country
        result = top_accounts_by_country(dummy_df, account_col="account", country_col="country", top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] <= 5


class TestPlatformLanguageBreakdown:
    def test_returns_dataframe(self, dummy_df):
        from contrastive_topic.analysis import platform_language_breakdown
        dummy_df["language"] = np.random.choice(["Arabic", "English", "Turkish"], len(dummy_df))
        result = platform_language_breakdown(dummy_df, platform_col="platform", language_col="language")
        assert isinstance(result, pd.DataFrame)


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:
    def setup_method(self):
        from contrastive_topic import Evaluation
        self.y_true = ["relevant", "irrelevant", "relevant", "relevant", "irrelevant"]
        self.y_pred = ["relevant", "irrelevant", "irrelevant", "relevant", "irrelevant"]
        self.ev = Evaluation(self.y_true, self.y_pred)

    def test_metrics_returns_dict(self):
        metrics = self.ev.metrics()
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_accuracy_correct(self):
        metrics = self.ev.metrics()
        assert metrics["accuracy"] == pytest.approx(0.8)

    def test_confusion_matrix_returns_figure(self):
        fig = self.ev.confusion_matrix()
        assert fig is not None

    def test_report_runs(self, capsys):
        self.ev.report()
        captured = capsys.readouterr()
        assert "relevant" in captured.out


# ── Translation utilities (no model download) ─────────────────────────────────

class TestTranslationUtils:
    def test_iter_batches(self):
        from contrastive_topic.translation import iter_batches
        data = list(range(10))
        batches = list(iter_batches(iter(data), batch_size=3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]

    def test_iter_df_as_dict(self):
        from contrastive_topic.translation import iter_df_as_dict
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rows = list(iter_df_as_dict(df))
        assert rows[0] == {"a": 1, "b": 3}
        assert len(rows) == 2


# ── FileHandler ───────────────────────────────────────────────────────────────

class TestFileHandler:
    def test_create_filename(self, tmp_path):
        from contrastive_topic import FileHandler
        fh = FileHandler(str(tmp_path))
        result = fh.create_filename("test.csv")
        assert "test.csv" in result

    def test_csv_roundtrip(self, tmp_path):
        from contrastive_topic import FileHandler
        fh = FileHandler(str(tmp_path))
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        fh.df_to_csv(df, "test.csv")
        loaded = pd.read_csv(fh.create_filename("test.csv"))
        assert list(loaded["x"]) == [1, 2, 3]

    def test_json_roundtrip(self, tmp_path):
        from contrastive_topic import FileHandler
        fh = FileHandler(str(tmp_path))
        data = {"key": "value", "num": 42}
        fh.save_json(data, "test.json")
        loaded = fh.read_json("test.json")
        assert loaded["key"] == "value"
