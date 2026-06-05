from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

DATASET_CHOICES = ("beauty", "electronics", "movielens")


@dataclass
class DatasetConfig:
    name: str
    source: str = "amazon"
    meta_hub: str | None = None
    meta_jsonl: str | None = None
    reviews_jsonl: str | None = None
    sample_nrows: int = 2_000_000

    def data_dir(self):
        return REPO_ROOT / "data" / self.name

    def reviews_csv(self):
        if self.source == "movielens":
            return self.data_dir() / "ratings.csv"
        return REPO_ROOT / "data" / f"{self.name}_reviews.csv"

    def movielens_raw_dir(self):
        return self.data_dir() / "raw" / "ml-1m"

    def ratings_dat(self):
        return self.movielens_raw_dir() / "ratings.dat"

    def movies_dat(self):
        return self.movielens_raw_dir() / "movies.dat"

    def links_csv(self):
        return self.movielens_raw_dir() / "links.csv"

    def tmdb_cache_path(self):
        return self.data_dir() / "tmdb_cache.json"

    def train_csv(self):
        p = self.data_dir() / "train.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = REPO_ROOT / "data" / "train.csv"
            if legacy.is_file():
                return legacy
        return p

    def test_csv(self):
        p = self.data_dir() / "test.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = REPO_ROOT / "data" / "test.csv"
            if legacy.is_file():
                return legacy
        return p

    def meta_csv(self):
        p = self.data_dir() / "meta.csv"
        if p.is_file():
            return p
        if self.name == "beauty":
            for legacy in (REPO_ROOT / "data" / "beauty_meta.csv", REPO_ROOT / "beauty_data.csv"):
                if legacy.is_file():
                    return legacy
        return p

    def emb_path(self):
        p = self.data_dir() / "item_embeddings.npy"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = REPO_ROOT / "data" / "item_embeddings.npy"
            if legacy.is_file():
                return legacy
        return p

    def asin2id_path(self):
        p = self.data_dir() / "asin2id.json"
        if p.is_file():
            return p
        if self.name == "beauty":
            legacy = REPO_ROOT / "data" / "asin2id.json"
            if legacy.is_file():
                return legacy
        return p

    def sasrec_txt(self):
        return REPO_ROOT / "baselines" / "SASRec.pytorch" / "python" / "data" / f"{self.name}.txt"

    def checkpoint_dir(self):
        p = REPO_ROOT / "checkpoints" / self.name
        if p.is_dir() and any(p.glob("coast*.pt")):
            return p
        if self.name == "beauty":
            legacy = REPO_ROOT / "checkpoints"
            if legacy.is_dir() and any(legacy.glob("coast*.pt")):
                return legacy
        return p

    def best_checkpoint_name(self, hybrid: bool) -> str:
        return "coast_hybrid_best.pt" if hybrid else "coast_best.pt"

    def train_log_path(self):
        return self.checkpoint_dir() / "train_log.json"

    def clcrec_data_name(self) -> str:
        return f"coast_{self.name}"

    def coast_train_defaults(self) -> dict:
        base = {
            "maxlen": 50,
            "early_stop_patience": 4,
            "min_epochs": 4,
            "early_stop_min_delta": 0.001,
        }
        if self.name == "beauty":
            return {
                **base,
                "num_epochs": 30,
                "dropout_rate": 0.3,
                "early_stop_patience": 3,
            }
        if self.name == "electronics":
            return {
                **base,
                "num_epochs": 30,
                "dropout_rate": 0.25,
            }
        if self.name == "movielens":
            return {
                **base,
                "num_epochs": 50,
                "dropout_rate": 0.3,
                "early_stop_patience": 5,
                "min_epochs": 5,
            }
        return {**base, "num_epochs": 20, "dropout_rate": 0.2}

    def results_path(self):
        return REPO_ROOT / "results" / f"{self.name}.json"


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
    meta_jsonl="raw/meta_categories/meta_Electronics.jsonl",
    reviews_jsonl="raw/review_categories/Electronics.jsonl",
)

MOVIELENS = DatasetConfig(
    name="movielens",
    source="movielens",
)

DATASETS = {c.name: c for c in (BEAUTY, ELECTRONICS, MOVIELENS)}


def get_dataset(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[key]
