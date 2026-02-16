"""Feature engineering pipeline for historical football match data.

This script extends the provided baseline by adding:
- input validation for required columns
- safer numeric coercion for mixed/dirty CSV values
- optional CLI arguments
- clear run summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "closing_ah",
    "opening_ah",
    "closing_odds_home",
    "opening_odds_home",
    "home_form_last5",
    "away_form_last5",
}

NUMERIC_COLUMNS = sorted(REQUIRED_COLUMNS)


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the input file is missing required fields."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(
            "Input dataset is missing required column(s): " f"{missing_columns}"
        )


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce expected numeric columns while preserving NaN for invalid values."""
    before_nulls = df[columns].isna().sum()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    after_nulls = df[columns].isna().sum()
    df.attrs["coerced_nulls"] = (after_nulls - before_nulls).clip(lower=0).to_dict()
    return df


def engineer_features(df: pd.DataFrame, trap_form_threshold: float = 5.0) -> pd.DataFrame:
    """Create engineered columns used for modeling and analysis."""
    # 1. Goal Difference
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    # 2. xG Differential
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    # 3. Line Movement Delta
    df["line_delta"] = df["closing_ah"] - df["opening_ah"]

    # 4. Odds Drift (Home)
    df["odds_drift_home"] = df["closing_odds_home"] - df["opening_odds_home"]

    # 5. Form Differential
    df["form_diff"] = df["home_form_last5"] - df["away_form_last5"]

    # 6. Asian Handicap Cover Result
    # NOTE: Binary output retained for compatibility with existing downstream usage.
    df["home_cover_result"] = ((df["goal_diff"] + df["closing_ah"]) > 0).astype("Int64")

    # 7. Over 2.5 Result
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["over25_result"] = (df["total_goals"] > 2.5).astype("Int64")

    # 8. Trap Signal Indicator
    trap_conditions = (
        (df["form_diff"] > trap_form_threshold)
        & (df["line_delta"] > 0)
        & (df["odds_drift_home"] > 0)
    )
    df["trap_signal"] = np.where(trap_conditions, 1, 0).astype("Int64")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature engineering for match data")
    parser.add_argument(
        "--input",
        default="555sport/data/historical_master.csv",
        help="Input CSV path (default: 555sport/data/historical_master.csv)",
    )
    parser.add_argument(
        "--output",
        default="555sport/data/historical_engineered.csv",
        help="Output CSV path (default: 555sport/data/historical_engineered.csv)",
    )
    parser.add_argument(
        "--trap-form-threshold",
        type=float,
        default=5.0,
        help="Minimum form differential used in trap_signal rule (default: 5.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_csv(input_path)
    validate_columns(data)
    data = coerce_numeric(data, NUMERIC_COLUMNS)
    data = engineer_features(data, trap_form_threshold=args.trap_form_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print("✅ Feature Engineering Completed")
    print(f"Rows processed: {len(data):,}")
    print(f"Saved to: {output_path}")
    coerced_nulls = data.attrs.get("coerced_nulls", {})
    newly_null_columns = {k: v for k, v in coerced_nulls.items() if v > 0}
    if newly_null_columns:
        formatted = ", ".join(
            f"{column}={count}" for column, count in sorted(newly_null_columns.items())
        )
        print(f"New NaN values from numeric coercion: {formatted}")


if __name__ == "__main__":
    main()
