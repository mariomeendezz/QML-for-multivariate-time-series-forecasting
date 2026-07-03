from pathlib import Path

import pandas as pd
from tabulate import tabulate


def print_tables(
    csv_path: str | Path = Path("results") / "lorenz_metrics.csv",
    decimals: int = 4
) -> None:
    """
    Read the metrics CSV and print one table for short-term and one for long-term forecasting.
    """
    # Read metrics CSV
    df = pd.read_csv(csv_path)

    # Format metric columns
    metrics = ["MAPE", "MAE", "RMSE"]
    df[metrics] = df[metrics].round(decimals)

    # Print one table per forecasting type
    for forecasting in ["short-term", "long-term"]:
        table = df[df["forecasting"] == forecasting][["model", *metrics]]

        print(f"\n{forecasting.upper()}")
        print(tabulate(table, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    print_tables()