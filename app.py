import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

DATA_PATH = "data/nyiso_combined.csv"


def find_column(columns, keywords):
    """
    Find the first column whose name contains all keywords (case insensitive).
    """
    for col in columns:
        name = col.lower()
        if all(k.lower() in name for k in keywords):
            return col
    raise ValueError(f"Could not find column with keywords: {keywords}")


@st.cache_data
def load_and_clean(path: str) -> pd.DataFrame:
    # Skip "==> file <==" markers that appear when CSVs are concatenated
    df = pd.read_csv(path, low_memory=False, comment="=")

    # Identify relevant columns by partial name matching
    time_col = find_column(df.columns, ["time"])
    zone_col = find_column(df.columns, ["name"])
    lmp_col = find_column(df.columns, ["lbmp"])
    cong_col = find_column(df.columns, ["congestion"])

    # Drop repeated header rows that may be embedded in the body
    df = df[~df[time_col].astype(str).str.contains("time stamp", case=False, na=False)]

    # Force numeric types for price related columns
    for col in [lmp_col, cong_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename to simple names
    df = df.rename(
        columns={
            time_col: "timestamp",
            zone_col: "zone",
            lmp_col: "lmp",
            cong_col: "congestion",
        }
    )

    # Parse timestamps and drop bad rows
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "lmp", "congestion", "zone"])

    # Create a cleaned zone key
    df["zone_key"] = df["zone"].str.upper()
    df["zone_key"] = df["zone_key"].str.replace("[^A-Z]", "", regex=True)

    return df


def analyze_zone(sub: pd.DataFrame, zone_key: str):
    x = sub["congestion"].values
    y = sub["lmp"].values

    if len(sub) < 2:
        return None

    # Correlation
    r = np.corrcoef(x, y)[0, 1]

    # Regression y = m x + b
    m, b = np.polyfit(x, y, 1)

    return {
        "zone": zone_key,
        "n": len(sub),
        "r": r,
        "m": m,
        "b": b,
    }


def main():
    st.title("NYISO Congestion vs LMP Viewer")

    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find data file at {DATA_PATH}")
        return

    df = load_and_clean(DATA_PATH)

    st.write(f"Total rows after cleaning: {len(df):,}")

    # Show available zones
    zone_keys = sorted(df["zone_key"].unique())
    st.sidebar.subheader("Zone selection")
    selected_zones = st.sidebar.multiselect(
        "Choose zones",
        zone_keys,
        default=[z for z in ["NYC", "WEST"] if z in zone_keys],
    )

    if not selected_zones:
        st.warning("Select at least one zone to display.")
        return

    # Option to filter to positive congestion only
    positive_only = st.sidebar.checkbox("Use only hours with positive congestion", value=True)

    df_sub = df[df["zone_key"].isin(selected_zones)].copy()
    if positive_only:
        df_sub = df_sub[df_sub["congestion"] > 0]

    st.write(f"Rows in current selection: {len(df_sub):,}")

    if df_sub.empty:
        st.warning("No data for this selection. Try turning off the positive congestion filter or choosing different zones.")
        return

    # Show a small preview of the data
    st.subheader("Sample of filtered data")
    st.dataframe(df_sub[["timestamp", "zone_key", "lmp", "congestion"]].head())

    # Scatter plot with regression line per zone
    st.subheader("Congestion vs LMP")

    fig, ax = plt.subplots(figsize=(9, 6))

    stats_rows = []

    for zone_key in selected_zones:
        sub = df_sub[df_sub["zone_key"] == zone_key]
        if sub.empty:
            continue

        stats = analyze_zone(sub, zone_key)
        if stats is None:
            continue

        stats_rows.append(stats)

        # Scatter points
        ax.scatter(sub["congestion"], sub["lmp"], alpha=0.25, label=f"{zone_key}")

        # Regression line
        x_vals = np.linspace(sub["congestion"].min(), sub["congestion"].max(), 100)
        y_vals = stats["m"] * x_vals + stats["b"]
        ax.plot(x_vals, y_vals)

    ax.set_xlabel("Marginal Cost of Congestion ($/MWh)")
    ax.set_ylabel("Locational Marginal Price, LMP ($/MWh)")
    ax.set_title("Congestion vs LMP for selected zones")
    ax.legend()

    st.pyplot(fig)

    # Show stats table
    if stats_rows:
        st.subheader("Correlation and regression statistics")
        stats_df = pd.DataFrame(stats_rows)
        # Round for display
        stats_df["r"] = stats_df["r"].round(4)
        stats_df["m"] = stats_df["m"].round(4)
        stats_df["b"] = stats_df["b"].round(4)
        st.table(stats_df.set_index("zone"))


if __name__ == "__main__":
    main()
