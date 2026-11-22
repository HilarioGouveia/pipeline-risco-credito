import shutil
from pathlib import Path
import pandas as pd
from src.config import BRONZE_CSV


def ensure_bronze_dataset(fallback_paths: list[Path] | None = None) -> Path:
    """Garante que o CSV bruto esteja na camada Bronze."""
    if BRONZE_CSV.exists():
        return BRONZE_CSV

    candidates = fallback_paths or [
        Path("credit_risk_dataset.csv"),
        Path("../credit_risk_dataset.csv"),
    ]
    for path in candidates:
        if path.exists():
            BRONZE_CSV.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, BRONZE_CSV)
            return BRONZE_CSV

    raise FileNotFoundError(
        f"Dataset não encontrado em {BRONZE_CSV}. Coloque 'credit_risk_dataset.csv' na pasta data/bronze/."
    )


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
