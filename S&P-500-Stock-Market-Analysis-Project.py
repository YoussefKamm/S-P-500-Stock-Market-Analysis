# _________________________________________________
# Include libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import Output, VBox, HBox, Dropdown, ToggleButton, Layout, HTML

# Use inline mode
%matplotlib inline




# _________________________________________________
# 1) Data Collection: scrape S&P 500 symbols from Wikipedia
def scrape_sp500_symbols(url: str) -> pd.DataFrame:
    """
    Scrapes S&P 500 symbols and company names from a Wikipedia table.

    Args:
        url: Wikipedia URL containing the S&P 500 constituents table.

    Returns:
        DataFrame with columns: ["Symbol", "Name"]
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})

    symbols = []
    if table:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                symbol = cols[0].text.strip().replace(".", "-")  # Yahoo uses '-' instead of '.'
                name = cols[1].text.strip()
                symbols.append((symbol, name))

    return pd.DataFrame(symbols, columns=["Symbol", "Name"])


# _________________________________________________
# 2) Data Collection: download historical data safely (Yahoo Finance)
def fetch_history_safe(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical price data for a symbol from Yahoo Finance using yfinance.

    This function is defensive:
      - Attempts the requested date range first
      - If empty, tries maximum history then filters to the requested window
      - Returns an empty DataFrame on any failure (prevents pipeline crashes)

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        A DataFrame indexed by Date (if available), otherwise empty DataFrame.
    """
    try:
        t = yf.Ticker(symbol)

        # Try direct range
        data = t.history(start=start_date, end=end_date, auto_adjust=False)
        if data is not None and not data.empty:
            return data

        # Fallback: get full history then filter
        data = t.history(period="max", auto_adjust=False)
        if data is None or data.empty:
            return pd.DataFrame()

        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

        if data.empty:
            return pd.DataFrame()

        return data.set_index("Date")

    except Exception:
        return pd.DataFrame()


# _________________________________________________
# 3) Data Processing: helper to force any column to 1D
def to_1d_array(x, length: int):
    """
    Converts an input to a 1D numpy array (defensive helper for weird shapes).
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.reshape(-1)
    return np.full(length, arr)


# _________________________________________________
# 4) Data Processing: normalize yfinance output (consistent columns & types)
def normalize_price_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw Yahoo Finance data into consistent columns:
        Date, Open, High, Low, Close, Volume

    Handles MultiIndex columns and missing columns safely.

    Args:
        raw: raw DataFrame from yfinance.

    Returns:
        Clean DataFrame with standard schema.
    """
    df = raw.copy()

    # If yfinance returns MultiIndex columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    # Ensure 'Date' exists
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Ensure required columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    n = len(df)

    clean = pd.DataFrame({
        "Date": pd.to_datetime(to_1d_array(df["Date"], n), errors="coerce"),
        "Open": pd.to_numeric(to_1d_array(df["Open"], n), errors="coerce"),
        "High": pd.to_numeric(to_1d_array(df["High"], n), errors="coerce"),
        "Low": pd.to_numeric(to_1d_array(df["Low"], n), errors="coerce"),
        "Close": pd.to_numeric(to_1d_array(df["Close"], n), errors="coerce"),
        "Volume": pd.to_numeric(to_1d_array(df["Volume"], n), errors="coerce"),
    })

    return clean.dropna(subset=["Date"]).reset_index(drop=True)


# _________________________________________________
# 5) MAIN PIPELINE: build the project tables (symbols, historical prices, moving averages)
# NOTE: In notebooks, this block will run when the cell executes, but the variables remain available.
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    start_date = "2018-01-01"
    end_date = "2025-12-30"

    # --- (A) Symbols table ---
    symbols_df = scrape_sp500_symbols(url)
    symbols_df = symbols_df.sort_values(by="Symbol").reset_index(drop=True)
    symbols_df.insert(0, "ID", range(1, len(symbols_df) + 1))
    print("Symbols table created")

    # --- (B) Historical + metrics tables ---
    historical_tables = []
    metrics_tables = []

    total = len(symbols_df)

    for i, row in symbols_df.iterrows():
        stock_id = int(row["ID"])
        symbol = row["Symbol"]

        raw = fetch_history_safe(symbol, start_date, end_date)
        if raw.empty:
            continue

        data = normalize_price_data(raw)
        if data.empty:
            continue

        # Historical prices (raw OHLCV)
        hist = data.copy()
        hist.insert(0, "Symbol", symbol)
        hist.insert(0, "ID", stock_id)
        historical_tables.append(hist[["ID", "Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]])

        # Calculated metrics: rolling moving averages (1..50)
        met = pd.DataFrame({
            "ID": [stock_id] * len(data),
            "Symbol": [symbol] * len(data),
            "Date": data["Date"],
            "Close": data["Close"],
        })

        for w in range(1, 51):
            met[f"Moving-Avg-{w}"] = met["Close"].rolling(window=w, min_periods=w).mean()

        metrics_tables.append(met)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total} symbols...")

    # --- (C) Final DataFrames ---
    historical_prices_df = pd.concat(historical_tables, ignore_index=True)
    calculated_metrics_df = pd.concat(metrics_tables, ignore_index=True)

    print("Historical prices table created")
    print("Calculated metrics table created")


# _________________________________________________
# 6) DASHBOARD (Interactive Results with ipywidgets)
# NOTE:
# - The dashboard expects historical_prices_df and calculated_metrics_df to exist.
# - In Jupyter/Colab, running the cell will display the dashboard UI.

# Output areas (one per visual)
out_vol = Output()
out_cmp = Output()
out_ma = Output()

# Constant month labels (for monthly view)
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Get available symbols from the prepared DataFrames
all_symbols_hist = sorted(historical_prices_df["Symbol"].astype(str).unique().tolist())
all_symbols_met = sorted(calculated_metrics_df["Symbol"].astype(str).unique().tolist())

# Default selections
defaultA = "AAPL" if "AAPL" in all_symbols_hist else all_symbols_hist[0]
defaultB = "MSFT" if "MSFT" in all_symbols_hist else (all_symbols_hist[1] if len(all_symbols_hist) > 1 else all_symbols_hist[0])
defaultMA = defaultA if defaultA in all_symbols_met else all_symbols_met[0]

# Dashboard title (HTML)
dashboard_title = HTML(
    value="""
    <h1 style="text-align:center; margin-bottom:20px; font-family:Arial;">
        S&amp;P 500 Stock Market Analysis
    </h1>
    """
)

# Controls (grouped per visual)
dd_volume = Dropdown(options=all_symbols_hist, value=defaultA, description="Symbol:",
                     layout=Layout(width="240px"))
tgl_view = ToggleButton(value=False, description="Show Monthly",
                        layout=Layout(width="150px"))

dd_cmp1 = Dropdown(options=all_symbols_hist, value=defaultA, description="Symbol 1:",
                   layout=Layout(width="240px"))
dd_cmp2 = Dropdown(options=all_symbols_hist, value=defaultB, description="Symbol 2:",
                   layout=Layout(width="240px"))

dd_ma = Dropdown(options=all_symbols_met, value=defaultMA, description="Symbol:",
                 layout=Layout(width="240px"))


# _________________________________________________
# Plot functions (each draws inside its dedicated Output widget)
def draw_volume():
    """Top-left visual: trading volume aggregated by Year or Month for the selected symbol."""
    with out_vol:
        out_vol.clear_output()

        sym = dd_volume.value
        view = "MONTH" if tgl_view.value else "YEAR"

        d = historical_prices_df[historical_prices_df["Symbol"] == sym].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")

        fig, ax = plt.subplots(figsize=(7.0, 3.6))

        if d.empty:
            ax.set_title(f"{sym} – No data available")
        else:
            yearly = d.groupby(d["Date"].dt.year)["Volume"].sum()
            monthly = d.groupby(d["Date"].dt.month)["Volume"].sum().reindex(range(1, 13), fill_value=0)

            if view == "YEAR":
                ax.bar(yearly.index.astype(str), yearly.values)
                ax.set_title(f"{sym} – Trading Volume by Year")
                ax.set_xlabel("Year")
                ax.set_ylabel("Volume")
            else:
                ax.bar(month_names, monthly.values)
                ax.set_title(f"{sym} – Trading Volume by Month")
                ax.set_xlabel("Month")
                ax.set_ylabel("Volume")

        plt.tight_layout()
        plt.show()


def draw_compare():
    """Top-right visual: compare Open prices of two selected symbols (last ~500 days)."""
    with out_cmp:
        out_cmp.clear_output()

        s1 = dd_cmp1.value
        s2 = dd_cmp2.value

        def get_data(sym):
            x = historical_prices_df[historical_prices_df["Symbol"] == sym].copy()
            x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
            x = x.dropna(subset=["Date"]).sort_values("Date")
            return x.tail(500)

        d1 = get_data(s1)
        d2 = get_data(s2)

        fig, ax = plt.subplots(figsize=(7.0, 3.6))

        if not d1.empty:
            ax.plot(d1["Date"], d1["Open"], label=s1)
        else:
            ax.plot([], [], label=f"{s1} (no data)")

        if not d2.empty:
            ax.plot(d2["Date"], d2["Open"], label=s2)
        else:
            ax.plot([], [], label=f"{s2} (no data)")

        ax.set_title(f"Open Price Comparison: {s1} vs {s2}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Open Price")
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()


def draw_ma_bars():
    """Bottom visual: yearly average Close vs MA-20 and MA-50 (grouped bar chart)."""
    with out_ma:
        out_ma.clear_output()

        sym = dd_ma.value

        m = calculated_metrics_df[calculated_metrics_df["Symbol"] == sym].copy()
        m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
        m = m.dropna(subset=["Date"])

        fig, ax = plt.subplots(figsize=(14.5, 3.8))

        if m.empty:
            ax.set_title(f"{sym} – No data available")
            plt.tight_layout()
            plt.show()
            return

        # Aggregate to yearly means (keeps chart readable)
        m["Year"] = m["Date"].dt.year
        yearly = m.groupby("Year")[["Close", "Moving-Avg-20", "Moving-Avg-50"]].mean().tail(8)

        years = yearly.index.astype(str)
        x = np.arange(len(years))
        w = 0.25

        b1 = ax.bar(x - w, yearly["Close"], width=w, label="Close")
        b2 = ax.bar(x, yearly["Moving-Avg-20"], width=w, label="MA-20")
        b3 = ax.bar(x + w, yearly["Moving-Avg-50"], width=w, label="MA-50")

        # Add value labels above bars (readability)
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_title(f"{sym} – Yearly Average Price vs Moving Averages")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price")
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()


# _________________________________________________
# Callbacks: re-draw only the affected visual (efficient UI updates)
def on_vol_change(change):
    """Called when the volume symbol or toggle changes."""
    tgl_view.description = "Show Yearly" if tgl_view.value else "Show Monthly"
    draw_volume()

def on_cmp_change(change):
    """Called when either comparison symbol changes."""
    draw_compare()

def on_ma_change(change):
    """Called when the MA-bars symbol changes."""
    draw_ma_bars()


# Link widget events to callbacks
dd_volume.observe(on_vol_change, names="value")
tgl_view.observe(on_vol_change, names="value")

dd_cmp1.observe(on_cmp_change, names="value")
dd_cmp2.observe(on_cmp_change, names="value")

dd_ma.observe(on_ma_change, names="value")


# _________________________________________________
# Initial display (draw all charts once)
on_vol_change(None)
on_cmp_change(None)
on_ma_change(None)

# Dashboard layout: controls are grouped with their corresponding plot
block_left = VBox([HBox([dd_volume, tgl_view]), out_vol])
block_right = VBox([HBox([dd_cmp1, dd_cmp2]), out_cmp])
top_row = HBox([block_left, block_right], layout=Layout(gap="18px"))

bottom_block = VBox([HBox([dd_ma]), out_ma])

# Final display object (Jupyter/Colab renders this automatically)
VBox([dashboard_title, top_row, bottom_block])


symbols_df.to_csv("symbols_df.csv", index=False)
historical_prices_df.to_csv("historical_prices_df.csv", index=False)
calculated_metrics_df.to_csv("calculated_metrics_df.csv", index=False)