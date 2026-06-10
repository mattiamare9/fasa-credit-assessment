from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"

_TRAIN_COMPANIES = DATA_DIR / "train_companies.csv"
_TRAIN_NARRATIVES = DATA_DIR / "train_narratives.csv"
_TRAIN_OUTCOMES = DATA_DIR / "train_outcomes.csv"
_SCORING = DATA_DIR / "scoring_companies.csv"


def _check(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")


def load_train() -> pd.DataFrame:
    for p in (_TRAIN_COMPANIES, _TRAIN_NARRATIVES, _TRAIN_OUTCOMES):
        _check(p)

    companies = pd.read_csv(_TRAIN_COMPANIES)
    narratives = pd.read_csv(_TRAIN_NARRATIVES)
    outcomes = pd.read_csv(_TRAIN_OUTCOMES)

    df = (
        companies
        .merge(narratives[["company_id", "business_description"]], on="company_id", how="left")
        .merge(outcomes[["company_id", "defaulted"]], on="company_id", how="inner")
    )
    return df


def load_scoring() -> pd.DataFrame:
    _check(_SCORING)
    return pd.read_csv(_SCORING)
