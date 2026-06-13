from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"

DATASET_CHOICES = ("beauty", "electronics")


@dataclass
class DatasetConfig:
    name: str
    meta_hub: str | None = None
    meta_jsonl: str | None = None
    reviews_jsonl: str | None = None
    sample_nrows: int = 2_000_000

    def data_dir(self):
        return REPO_ROOT / "data" / self.name

    def reviews_csv(self):
        return REPO_ROOT / "data" / f"{self.name}_reviews.csv"

    def train_csv(self):
        return self.data_dir() / "train.csv"

    def test_csv(self):
        return self.data_dir() / "test.csv"

    def meta_csv(self):
        return self.data_dir() / "meta.csv"

    def emb_path(self):
        return self.data_dir() / "item_embeddings.npy"

    def asin2id_path(self):
        return self.data_dir() / "asin2id.json"

    def sasrec_txt(self):
        return REPO_ROOT / "baselines" / "SASRec.pytorch" / "python" / "data" / f"{self.name}.txt"

    def checkpoint_dir(self):
        return REPO_ROOT / "checkpoints" / self.name

    def best_checkpoint_name(self, hybrid: bool) -> str:
        return "coast_hybrid_best.pt" if hybrid else "coast_best.pt"

    def results_path(self):
        return REPO_ROOT / "results" / f"{self.name}.json"

    def coast_train_defaults(self) -> dict:
        if self.name == "beauty":
            return {"maxlen": 50, "num_epochs": 30, "dropout_rate": 0.3, "early_stop_patience": 3}
        if self.name == "electronics":
            return {"maxlen": 50, "num_epochs": 30, "dropout_rate": 0.25, "early_stop_patience": 4}
        return {"maxlen": 50, "num_epochs": 20, "dropout_rate": 0.2, "early_stop_patience": 4}


BEAUTY = DatasetConfig(
    name="beauty",
    meta_hub="smartcat/Amazon_Beauty_and_Personal_Care_2023",
    meta_jsonl="raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
    reviews_jsonl="raw/review_categories/Beauty_and_Personal_Care.jsonl",
)

ELECTRONICS = DatasetConfig(
    name="electronics",
    meta_jsonl="raw/meta_categories/meta_Electronics.jsonl",
    reviews_jsonl="raw/review_categories/Electronics.jsonl",
)

DATASETS = {c.name: c for c in (BEAUTY, ELECTRONICS)}


def get_dataset(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[key]
