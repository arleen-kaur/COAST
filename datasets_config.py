"""Dataset configs for the COAST pipeline (Amazon + MovieLens-1M)."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

DATASET_CHOICES = ("beauty", "electronics", "movielens")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str = "amazon"  # amazon | movielens
    meta_hub: str | None = None
    meta_jsonl: str | None = None
    reviews_jsonl: str | None = None
    sample_nrows: int = 2_000_000

    def data_dir(self) -> Path:
        return ROOT / "data" / self.name

    def reviews_csv(self) -> Path:
        if self.source == "movielens":
            return self.data_dir() / "ratings.csv"
        return ROOT / "data" / f"{self.name}_reviews.csv"

    def movielens_raw_dir(self) -> Path:
        return self.data_dir() / "raw" / "ml-1m"

    def ratings_dat(self) -> Path:
        return self.movielens_raw_dir() / "ratings.dat"

    def movies_dat(self) -> Path:
        return self.movielens_raw_dir() / "movies.dat"

    def links_csv(self) -> Path:
        return self.movielens_raw_dir() / "links.csv"

    def tmdb_cache_path(self) -> Path:
        return self.data_dir() / "tmdb_cache.json"

    def train_csv(self) -> Path:
        p = self.data_dir() / "train.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = ROOT / "data" / "train.csv"
            if legacy.is_file():
                return legacy
        return p

    def test_csv(self) -> Path:
        p = self.data_dir() / "test.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = ROOT / "data" / "test.csv"
            if legacy.is_file():
                return legacy
        return p

    def meta_csv(self) -> Path:
        p = self.data_dir() / "meta.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            for legacy in (ROOT / "data" / "beauty_meta.csv", ROOT / "beauty_data.csv"):
                if legacy.is_file():
                    return legacy
        return p

    def emb_path(self) -> Path:
        p = self.data_dir() / "item_embeddings.npy"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = ROOT / "data" / "item_embeddings.npy"
            if legacy.is_file():
                return legacy
        return p

    def asin2id_path(self) -> Path:
        p = self.data_dir() / "asin2id.json"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = ROOT / "data" / "asin2id.json"
            if legacy.is_file():
                return legacy
        return p

    def sasrec_txt(self) -> Path:
        return ROOT / "baselines" / "SASRec.pytorch" / "python" / "data" / f"{self.name}.txt"

    def checkpoint_dir(self) -> Path:
        p = ROOT / "checkpoints" / self.name
        if p.is_dir() and any(p.glob("coast*.pt")):
            return p
        if self.name == "beauty":
            legacy = ROOT / "checkpoints"
            if legacy.is_dir() and any(legacy.glob("coast*.pt")):
                return legacy
        return p

    def best_checkpoint_name(self, hybrid: bool) -> str:
        return "coast_hybrid_best.pt" if hybrid else "coast_best.pt"

    def train_log_path(self) -> Path:
        return self.checkpoint_dir() / "train_log.json"

    def coast_train_defaults(self) -> dict:
        """Recommended hyperparameters (early stopping enabled in train.py)."""
        if self.name == "movielens":
            return {"num_epochs": 50, "dropout_rate": 0.3, "maxlen": 50}
        return {"num_epochs": 20, "dropout_rate": 0.2, "maxlen": 50}

    def results_path(self) -> Path:
        return ROOT / "results" / f"{self.name}.json"


BEAUTY = DatasetConfig(
    name="beauty",
    source="amazon",
    meta_hub="smartcat/Amazon_Beauty_and_Personal_Care_2023",
    meta_jsonl="raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
    reviews_jsonl="raw/review_categories/Beauty_and_Personal_Care.jsonl",
)

ELECTRONICS = DatasetConfig(
    name="electronics",
    source="amazon",
    meta_hub=None,
    meta_jsonl="raw/meta_categories/meta_Electronics.jsonl",
    reviews_jsonl="raw/review_categories/Electronics.jsonl",
    sample_nrows=2_000_000,
)

MOVIELENS = DatasetConfig(
    name="movielens",
    source="movielens",
    sample_nrows=10_000_000,
)

DATASETS = {c.name: c for c in (BEAUTY, ELECTRONICS, MOVIELENS)}


def get_dataset(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[key]
