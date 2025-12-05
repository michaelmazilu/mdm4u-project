import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/nyiso_combined.csv"


def find_column(columns, keywords):
    """
    Find the first column whose name contains all keywords (case-insensitive).
    """
    for col in columns:
        name = col.lower()
        if all(k.lower() in name for k in keywords):
            return col
    raise ValueError(f"Could not find column with keywords: {keywords}")


def load_and_clean(path: str) -> pd.DataFrame:
    # Skip "==> file <==" markers that were introduced when the CSVs were concatenated
    df = pd.read_csv(path, comment="=")

    print("Original columns:")
    print(list(df.columns))
    print()

    # Infer column names from partial matches so it works even if NYISO changes labels slightly
    time_col = find_column(df.columns, ["time"])
    zone_col = find_column(df.columns, ["name"])          # zone name
    lmp_col = find_column(df.columns, ["lbmp"])           # total LMP
    cong_col = find_column(df.columns, ["congestion"])    # congestion component

    # Drop any stray header rows left in the body of the file
    df = df[~df[time_col].astype(str).str.contains("time stamp", case=False, na=False)]

    # Rename to simple standard names
    df = df.rename(
        columns={
            time_col: "timestamp",
            zone_col: "zone",
            lmp_col: "lmp",
            cong_col: "congestion",
        }
    )

    # Parse timestamps and numerics, then drop rows that fail
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
    df["congestion"] = pd.to_numeric(df["congestion"], errors="coerce")
    df = df.dropna(subset=["timestamp", "lmp", "congestion", "zone"])

    # Clean zone names into a simple key, for example:
    # "N.Y.C." -> "NYC", "WEST" -> "WEST"
    df["zone_key"] = df["zone"].str.upper()
    df["zone_key"] = df["zone_key"].str.replace("[^A-Z]", "", regex=True)

    print("Unique zone keys found:")
    print(sorted(df["zone_key"].unique()))
    print()

    return df


def analyze_zone(df: pd.DataFrame, zone_key: str):
    sub = df[df["zone_key"] == zone_key].copy()

    if sub.empty:
        print(f"No rows found for zone_key={zone_key}")
        return None

    x = sub["congestion"].values
    y = sub["lmp"].values

    # Correlation
    r = np.corrcoef(x, y)[0, 1]

    # Simple linear regression y = m x + b
    m, b = np.polyfit(x, y, 1)

    print(f"Zone {zone_key}:")
    print(f"  Number of observations: {len(sub)}")
    print(f"  Correlation (r) between congestion and LMP: {r:.4f}")
    print(f"  Regression line: LMP = {m:.4f} * Congestion + {b:.4f}")
    print()

    return sub, m, b


def make_scatter_with_regression(df: pd.DataFrame):
    # Focus on NYC and WEST
    target_zones = ["NYC", "WEST"]
    df_sub = df[df["zone_key"].isin(target_zones)].copy()

    if df_sub.empty:
        print("No data for NYC or WEST after filtering. Check zone_key values above.")
        return

    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for zone_key in target_zones:
        sub, m, b = analyze_zone(df_sub, zone_key)
        if sub is None:
            continue

        # Scatter points
        ax.scatter(sub["congestion"], sub["lmp"], alpha=0.25, label=f"{zone_key}")

        # Regression line
        x_vals = np.linspace(sub["congestion"].min(), sub["congestion"].max(), 100)
        y_vals = m * x_vals + b
        ax.plot(x_vals, y_vals)

    ax.set_xlabel("Marginal Cost of Congestion ($/MWh)")
    ax.set_ylabel("Locational Marginal Price, LMP ($/MWh)")
    ax.set_title("Congestion vs LMP for NYC and WEST")
    ax.legend()

    out_path = os.path.join("figures", "congestion_vs_lmp_nyc_west.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved scatterplot with regression to {out_path}")


def main():
    print(f"Loading data from {DATA_PATH}")
    df = load_and_clean(DATA_PATH)

    print(f"Total rows after cleaning: {len(df)}")
    print()

    make_scatter_with_regression(df)


if __name__ == "__main__":
    main()
