"""Amazon category configs for the COAST pipeline."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    meta_hub: str | None
    meta_jsonl: str
    reviews_jsonl: str
    sample_nrows: int = 2_000_000

    def data_dir(self) -> Path:
        return ROOT / "data" / self.name

    def reviews_csv(self) -> Path:
        return ROOT / "data" / f"{self.name}_reviews.csv"

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


BEAUTY = DatasetConfig(
    name="beauty",
    meta_hub="smartcat/Amazon_Beauty_and_Personal_Care_2023",
    meta_jsonl="raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
    reviews_jsonl="raw/review_categories/Beauty_and_Personal_Care.jsonl",
)

ELECTRONICS = DatasetConfig(
    name="electronics",
    meta_hub=None,
    meta_jsonl="raw/meta_categories/meta_Electronics.jsonl",
    reviews_jsonl="raw/review_categories/Electronics.jsonl",
    sample_nrows=2_000_000,
)

DATASETS = {c.name: c for c in (BEAUTY, ELECTRONICS)}


def get_dataset(name: str) -> DatasetConfig:
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[key]
