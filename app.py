import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="NYISO Congestion vs LMP",
    page_icon="📈",
    layout="wide",
)

# Enforce light plot styling and dark text
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.edgecolor": "black",
    }
)

# Light custom styling to tighten spacing and add a clear visual hierarchy
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&display=swap');
        html, body { background: #ffffff !important; color: #111111; font-family: "Lexend", "Segoe UI", sans-serif; }
        .stApp { background: #ffffff !important; color: #111111; }
        .main { padding: 1.5rem 3rem; background: #ffffff !important; color: #111111; }
        section[data-testid="stSidebar"] { min-width: 220px; max-width: 240px; }
        section[data-testid="stSidebar"] > div {
            background: #f0f1f5;
            color: #111111;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        section[data-testid="stSidebar"] * { color: #111111 !important; }
        /* Multiselect styling (input + dropdown) */
        .stMultiSelect div[role="combobox"] {
            background: #f0f1f5 !important;
            border: none !important;
            color: #111111 !important;
        }
        .stMultiSelect div[role="combobox"] * {
            color: #111111 !important;
            background: #f0f1f5 !important;
        }
        /* BaseWeb select/popover overrides */
        div[data-baseweb="select"] {
            background: #f0f1f5 !important;
            color: #111111 !important;
            border-color: transparent !important;
        }
        div[data-baseweb="select"] * {
            color: #111111 !important;
            background: #f0f1f5 !important;
        }
        div[data-baseweb="popover"] {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #cdd4da !important;
            box-shadow: 0 8px 18px rgba(0,0,0,0.08);
        }
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"] {
            background: #ffffff !important;
            color: #111111 !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover {
            background: #f2f4f7 !important;
        }
        /* Multi-select chips and input area */
        .stMultiSelect div[data-baseweb="tag"] {
            background: #f0f1f5 !important;
            color: #111111 !important;
            border: none !important;
            box-shadow: 0 0 0 1px #cdd4da inset;
        }
        .stMultiSelect div[data-baseweb="tag"] * {
            color: #111111 !important;
            background: #f0f1f5 !important;
        }
        .stMultiSelect svg { color: #111111 !important; fill: #111111 !important; }
        .stMultiSelect label { color: #111111 !important; }
        /* Force the multiselect input container itself to be light */
        .stMultiSelect div[data-baseweb="select"] > div {
            background: #f0f1f5 !important;
            color: #111111 !important;
            border-color: #cdd4da !important;
        }
        .stMultiSelect div[data-baseweb="select"] div {
            background: #f0f1f5 !important;
            color: #111111 !important;
        }
        /* Root wrapper of the multi-select control */
        .stMultiSelect > div {
            background: #f0f1f5 !important;
        }
        .stMultiSelect > div > div {
            background: #f0f1f5 !important;
            border: none !important;
        }
        /* Info banner hidden when empty */
        .info-banner:empty { display: none !important; }
        /* Hide Streamlit top toolbar/header bar */
        header, [data-testid="stHeader"], div[data-testid="stToolbar"], .stAppToolbar { display: none !important; height: 0 !important; visibility: hidden !important; }
        .page-wrap { max-width: 1200px; margin: 0 auto; }
        .app-hero h1 { font-size: 3rem; line-height: 1.05; margin-bottom: 0.4rem; color: #111111; letter-spacing: -0.5px; }
        .app-hero h1 strong { font-weight: 800; color: #111111; }
        .app-hero p { color: ##5e5e5e; margin-top: 0; max-width: 720px; font-size: 1.05rem; }
        hr.divider { border: none; border-top: 1px solid #e0dfd6; margin: 1.5rem 0; }
        .stat-card {
            background: #ffffff;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid #e0dfd6;
        }
        .stat-card h4 { margin: 0 0 0.35rem 0; font-size: 0.9rem; color: #3a3a3a; }
        .stat-card .value { font-size: 1.4rem; font-weight: 700; color: #111111; }
        .info-banner {
            background: #ffffff;
            color: #ffffff;
            padding: 0.85rem 1rem;
            border-radius: 10px;
            border: 1px solid #e0dfd6;
        }
        .section-title { font-size: 1.5rem; margin: 1rem 0 0.5rem 0; }
        /* Force dataframes/tables to be light */
        .stDataFrame, .stTable {
            background: #ffffff !important;
            color: #111111 !important;
        }
        .stDataFrame [data-baseweb="table"], .stDataFrame table, .stTable table,
        div[data-testid="stDataFrame"] table {
            background: #ffffff !important;
            color: #111111 !important;
        }
        /* Add visible gridlines to Streamlit tables */
        .stTable table {
            border-collapse: collapse !important;
            width: 100%;
            border: 1px solid #e0dfd6 !important;
        }
        .stTable table th, .stTable table td {
            border: 1px solid #e0dfd6 !important;
            padding: 6px 8px !important;
        }
        .stTable table thead tr,
        .stTable table thead th,
        .stTable table thead th:empty {
            background: #f7f8fb !important;  /* keep header a single color */
        }
        .stTable table thead th {
            font-weight: 700 !important;
        }
        .stDataFrame [data-baseweb="table"] *, .stDataFrame table * , .stTable table *,
        div[data-testid="stDataFrame"] * {
            color: #111111 !important;
            background-color: #ffffff !important;
        }
        div[data-testid="stDataFrame"] {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #e0dfd6;
            border-radius: 8px;
        }
        /* Generic table text force */
        table, th, td {
            color: #111111 !important;
        }
        /* Custom lightweight table for the sample preview */
        .light-table table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            color: #111111;
            font-size: 0.95rem;
        }
        .light-table th, .light-table td {
            border: 1px solid #e0dfd6;
            padding: 6px 8px;
            text-align: left;
        }
        .light-table th {
            font-weight: 700;
            background: #ffffff;
        }
        /* Make matplotlib/pyplot images scale with the container (browser zoom/resizes) */
        [data-testid="stImage"] img {
            width: 100% !important;
            height: auto !important;
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def remove_iqr_outliers(df: pd.DataFrame, zone_col: str, value_col: str, mult: float = 1.5) -> pd.DataFrame:
    """Per-zone IQR filter (keeps rows within [Q1 - mult*IQR, Q3 + mult*IQR])."""
    grouped = df.groupby(zone_col)[value_col]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - mult * iqr
    upper = q3 + mult * iqr
    mask = (df[value_col] >= lower) & (df[value_col] <= upper)
    return df[mask].copy()


def main():
    st.markdown(
        """
        <div class="page-wrap">
            <div class="app-hero">
                <h1><strong>NYISO</strong> Congestion vs LMP</h1>
                <p>Explore how congestion costs shape LMP across every zone. Filter quickly, see correlations instantly, and keep the signal clean.</p>
            </div>
            <hr class="divider" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find data file at {DATA_PATH}")
        return

    df = load_and_clean(DATA_PATH)
    total_rows = len(df)

    # Show available zones
    zone_keys = sorted(df["zone_key"].unique())
    st.sidebar.subheader("Zone selection")
    selected_zones = st.sidebar.multiselect(
        "Choose zones",
        zone_keys,
        default=[z for z in ["NYC", "WEST"] if z in zone_keys],
    )
    scatter_layout = st.sidebar.radio(
        "Scatter layout",
        options=["Split per zone", "Single combined"],
        index=0,
        help="Separate charts reduce overlap; switch to single combined if you prefer one view.",
    )
    exclude_outliers = st.sidebar.checkbox("Exclude LMP outliers (1.5×IQR per zone)", value=True)

    if not selected_zones:
        st.warning("Select at least one zone to display.")
        return

    # Option to filter to positive congestion only
    positive_only = st.sidebar.checkbox("Use only hours with positive congestion", value=True)

    df_sub = df[df["zone_key"].isin(selected_zones)].copy()
    if positive_only:
        df_sub = df_sub[df_sub["congestion"] > 0]
    if exclude_outliers:
        df_sub = remove_iqr_outliers(df_sub, "zone_key", "lmp")

    points_on_chart = len(df_sub)

    # Stats row
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f'<div class="stat-card"><h4>Total rows after cleaning</h4><div class="value">{total_rows:,}</div></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<div class="stat-card"><h4>Rows in current selection</h4><div class="value">{points_on_chart:,}</div></div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div class="stat-card"><h4>Positive congestion only</h4><div class="value">{"Yes" if positive_only else "No"}</div></div>',
                unsafe_allow_html=True,
            )

    if df_sub.empty:
        st.warning("No data for this selection. Try turning off the positive congestion filter or choosing different zones.")
        return

    # Show how many points will be plotted on the chart (total and per zone)
    per_zone_counts = df_sub["zone_key"].value_counts()
    per_zone_text = ", ".join(f"{zone}: {per_zone_counts.get(zone, 0):,}" for zone in selected_zones)
    # Skip the old info banner to reduce clutter

    # Show a small preview of the data (rendered as a lightweight HTML table to force light theme)
    st.markdown('<div class="section-title">Sample of filtered data</div>', unsafe_allow_html=True)
    sample_df = df_sub[["timestamp", "zone_key", "lmp", "congestion"]].head().reset_index(drop=True)
    sample_html = sample_df.to_html(index=False, classes="light-table")
    st.markdown(f'<div class="light-table">{sample_html}</div>', unsafe_allow_html=True)

    # Scatter plot with regression line per zone
    st.markdown('<div class="section-title">Congestion vs LMP</div>', unsafe_allow_html=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    zone_colors = {
        "NYC": "#d62828",   # red
        "WEST": "#1d4ed8",  # blue
    }
    stats_rows = []
    zone_plots = []

    for zone_key in selected_zones:
        sub = df_sub[df_sub["zone_key"] == zone_key]
        if sub.empty:
            continue

        stats = analyze_zone(sub, zone_key)
        if stats is None:
            continue

        stats_rows.append(stats)
        zone_plots.append((zone_key, sub, stats))

    if not zone_plots:
        st.warning("No data available for the selected zones.")
    else:
        if scatter_layout == "Single combined" or len(zone_plots) == 1:
            fig, ax = plt.subplots(figsize=(9, 6))
            for zone_key, sub, stats in zone_plots:
                color = zone_colors.get(zone_key, None)
                ax.scatter(sub["congestion"], sub["lmp"], alpha=0.25, label=f"{zone_key}", color=color)
                x_vals = np.linspace(sub["congestion"].min(), sub["congestion"].max(), 100)
                y_vals = stats["m"] * x_vals + stats["b"]
                ax.plot(x_vals, y_vals, color=color)

            ax.set_xlabel("Marginal Cost of Congestion ($/MWh)")
            ax.set_ylabel("Locational Marginal Price, LMP ($/MWh)")
            ax.set_title("Congestion vs LMP")
            ax.legend()

            cols_chart = st.columns([1, 6, 1])
            with cols_chart[1]:
                st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Split view: one chart per selected zone to reduce overlap.")
            cols_per_row = 2 if len(zone_plots) > 1 else 1
            for idx in range(0, len(zone_plots), cols_per_row):
                row_items = zone_plots[idx : idx + cols_per_row]
                cols = st.columns(len(row_items))
                for (zone_key, sub, stats), col in zip(row_items, cols):
                    with col:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        color = zone_colors.get(zone_key, None)
                        ax.scatter(sub["congestion"], sub["lmp"], alpha=0.25, label=f"{zone_key}", color=color)
                        x_vals = np.linspace(sub["congestion"].min(), sub["congestion"].max(), 100)
                        y_vals = stats["m"] * x_vals + stats["b"]
                        ax.plot(x_vals, y_vals, color=color)
                        ax.set_xlabel("Marginal Cost of Congestion ($/MWh)")
                        ax.set_ylabel("Locational Marginal Price, LMP ($/MWh)")
                        ax.set_title(f"Congestion vs LMP — {zone_key}")
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)

    # Box plot comparing LMP distributions for selected zones (uses full dataset; optional IQR filtering)
    st.markdown('<div class="section-title">LMP distribution (selected zones)</div>', unsafe_allow_html=True)
    box_df = df[df["zone_key"].isin(selected_zones)].copy()
    if exclude_outliers:
        box_df = remove_iqr_outliers(box_df, "zone_key", "lmp")
    ordered_zones = selected_zones
    box_data = []
    box_labels = []
    per_zone_stats = []
    for z in ordered_zones:
        series = box_df[box_df["zone_key"] == z]["lmp"].dropna()
        if not series.empty:
            box_data.append(series)
            box_labels.append(z)
            per_zone_stats.append(
                {
                    "zone": z,
                    "median": float(series.median()),
                    "q1": float(series.quantile(0.25)),
                    "q3": float(series.quantile(0.75)),
                }
            )

    if not box_data:
        st.info("LMP box plot not available because NYC/WEST data is missing.")
    else:
        combined_lmp = pd.concat(box_data)
        min_lmp = np.nanmin(combined_lmp)
        max_lmp = np.nanmax(combined_lmp)
        x_min, x_max = -150, 250  # keep both tails in view; shows ticks at -100 and 200
        span = max(x_max - x_min, 1e-6)

        fig_box, ax_box = plt.subplots(figsize=(7.5, 3.75))
        bp = ax_box.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showfliers=False,  # hide raw dots for a clean box-only view
            vert=False,  # horizontal orientation
        )
        default_box_color = "#6b7280"
        colors_for_boxes = [zone_colors.get(z, default_box_color) for z in box_labels]
        for patch, color in zip(bp["boxes"], colors_for_boxes):
            patch.set(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.5)
        for median, color in zip(bp["medians"], colors_for_boxes):
            median.set(color=color, linewidth=2)
        for whisker in bp["whiskers"] + bp["caps"]:
            whisker.set(color="#555555", linewidth=1.2)
        ax_box.axvline(0, color="#888888", linestyle="--", linewidth=1)
        ax_box.set_xlim(x_min, x_max)
        ax_box.set_xticks(np.arange(-150, 251, 50))
        ax_box.set_xlabel("Locational Marginal Price, LMP ($/MWh)")
        ax_box.set_ylabel("Zone")
        ax_box.set_title("LMP distribution for NYC vs WEST")

        # Annotate medians and IQR to make interpretation explicit
        for idx, stats in enumerate(per_zone_stats):
            y_pos = idx + 1  # matplotlib boxplot positions start at 1
            x_pos = x_min + 0.02 * span
            text = f"Med {stats['median']:.1f}; IQR {stats['q1']:.1f}–{stats['q3']:.1f}"
            ax_box.text(
                x_pos,
                y_pos + 0.2,
                text,
                fontsize=9,
                color=zone_colors.get(stats["zone"], "#222222"),
                va="center",
                ha="left",
            )

        cols_box = st.columns([1, 6, 1])
        with cols_box[1]:
            st.pyplot(fig_box, use_container_width=True)
        outlier_note = "Outliers removed via 1.5×IQR per zone." if exclude_outliers else "Outliers included."
        st.caption(
            "Boxes show IQR (25th–75th percentile) per selected zone, the line marks the median. "
            "Axis fixed to -150 to 250 so both tails (incl. -100 and 200 marks) stay visible; extreme spikes may exist outside. "
            f"{outlier_note}"
        )

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
