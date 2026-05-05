import time
import warnings
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

START_TIME = time.time()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TICKERS = {
    "Tech":        ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Healthcare":  ["JNJ",  "PFE",  "UNH",   "ABBV"],
    "Utilities":   ["NEE",  "DUK",  "SO",    "AEP"],
    "Commodities": ["XOM",  "CVX",  "COP",   "SLB"],
    "Finance":     ["JPM",  "BAC",  "GS",    "WFC"]
}

ALL_TICKERS     = [t for s in TICKERS.values() for t in s]
SECTOR_MAP      = {t: s for s, ticks in TICKERS.items() for t in ticks}

# ── Dynamic dates: always rolling 1 year up to today ──
END_DATE        = datetime.today().strftime("%Y-%m-%d")
START_DATE      = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

print(f"Date range: {START_DATE} → {END_DATE}")

NUM_STOCKS      = 10
MIN_WEIGHT      = 0.05
MAX_WEIGHT      = 0.30
RISK_FREE_RATE  = 0.05 / 252
WINDOW          = 5
INITIAL_CAPITAL = 10_000.0
MA_SHORT        = 5
MA_LONG         = 20

# ── CNN Fear & Greed threshold ──
# Score 0–25  → Extreme Fear → Hold Cash
# Score 26–100 → Normal     → Run MPT
CNN_FEAR_THRESHOLD = 25

OUTPUT_DIR      = "outputs/"


# ─────────────────────────────────────────────
# Data Download
# ─────────────────────────────────────────────

def download_prices():
    import time as t
    print("Downloading stock data...")
    all_data = []
    for ticker in ALL_TICKERS:
        for attempt in range(5):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE,
                                 progress=False, auto_adjust=True, threads=False)
                if df is not None and not df.empty:
                    close = df["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close = close.dropna()
                    close.index = pd.to_datetime(close.index).normalize()
                    close.name = ticker
                    all_data.append(close)
                    print(f"  Got {ticker}")
                    break
            except Exception:
                pass
            t.sleep(3)
        else:
            print(f"  FAILED {ticker}")

    if not all_data:
        raise ValueError("No data downloaded.")

    prices = pd.concat(all_data, axis=1).dropna()
    print(f"Price matrix shape: {prices.shape}")
    return prices


def download_spy():
    df = yf.download("SPY", start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError("SPY download failed.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).normalize()
    return close


def download_cnn_fear_greed():
    """
    Download CNN Fear & Greed Index historical data.
    Uses the fear-greed package which pulls from CNN's internal API.

    CNN Fear & Greed Index (0-100):
      0  – 25  → Extreme Fear  → Portfolio holds cash
      26 – 45  → Fear
      46 – 55  → Neutral
      56 – 75  → Greed
      76 – 100 → Extreme Greed

    Based on 7 market indicators:
      1. Market Momentum (S&P 500 vs 125-day MA)
      2. Stock Price Strength (52-week highs vs lows)
      3. Stock Price Breadth (McClellan Volume Summation)
      4. Put/Call Options ratio
      5. Market Volatility (VIX vs 50-day MA)
      6. Junk Bond Demand (yield spread)
      7. Safe Haven Demand (stocks vs bonds)
    """
    print("  Downloading CNN Fear & Greed Index...")
    try:
        import fear_greed
        history = fear_greed.get_history(last="365")

        records = []
        for point in history:
            records.append({
                "date":   pd.to_datetime(point.date).tz_localize(None).normalize(),
                "score":  point.score,
                "rating": point.rating
            })

        df = pd.DataFrame(records).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        current_score  = fear_greed.get_score()
        current_rating = fear_greed.get_rating()

        print(f"  CNN F&G Score: {current_score:.1f} ({current_rating})")
        print(f"  Score range (1yr): {df['score'].min():.1f} – {df['score'].max():.1f}")
        return df, current_score, current_rating

    except Exception as e:
        print(f"  CNN Fear & Greed failed: {e}")
        print("  Falling back to VIX as fear proxy...")
        return _fallback_vix()


def _fallback_vix():
    """
    Fallback to VIX if CNN API is unavailable.
    Converts VIX to a 0-100 fear score (inverted — high VIX = low score = more fear).
    VIX 10 → score ~90 (extreme greed)
    VIX 25 → score ~50 (neutral)
    VIX 45 → score ~10 (extreme fear)
    """
    print("  Using VIX fallback...")
    df_vix = yf.download("^VIX", start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False, threads=False)
    close = df_vix["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).normalize()

    # Convert VIX to 0-100 fear score (inverted)
    score = 100 - ((close - 10) / (80 - 10) * 100).clip(0, 100)

    df = pd.DataFrame({
        "score":  score.values,
        "rating": score.apply(lambda s:
            "extreme fear" if s <= 25 else
            "fear"         if s <= 45 else
            "neutral"      if s <= 55 else
            "greed"        if s <= 75 else
            "extreme greed"
        )
    }, index=close.index)

    current_score  = float(df["score"].iloc[-1])
    current_rating = df["rating"].iloc[-1]
    return df, current_score, current_rating


def compute_returns(prices):
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────
# Portfolio Metrics
# ─────────────────────────────────────────────

def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights, np.dot(cov_matrix * 252, weights)))
    sharpe = (ret - RISK_FREE_RATE * 252) / vol if vol > 0 else 0
    return ret, vol, sharpe


# ─────────────────────────────────────────────
# Stock Selection — Top 2 Sharpe per Sector
# ─────────────────────────────────────────────

def select_top_stocks(returns):
    selected = []
    for sector, tickers in TICKERS.items():
        available = [t for t in tickers if t in returns.columns]
        sharpes = {}
        for t in available:
            r = returns[t]
            sharpes[t] = (r.mean() - RISK_FREE_RATE) / r.std() if r.std() > 0 else 0
        top2 = sorted(sharpes, key=sharpes.get, reverse=True)[:2]
        selected.extend(top2)
    return selected[:NUM_STOCKS]


# ─────────────────────────────────────────────
# Optimization — Maximum Sharpe Ratio (SLSQP)
# ─────────────────────────────────────────────

def optimize_portfolio(returns_subset):
    n = len(returns_subset.columns)
    if n < 2:
        return None, None, None, None

    mean_ret = returns_subset.mean()
    cov_mat  = returns_subset.cov()

    def neg_sharpe(w):
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        return -s

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))
    init_w      = np.full(n, 1.0 / n)

    result = minimize(neg_sharpe, init_w, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 500})

    if result.success:
        w = np.clip(result.x, MIN_WEIGHT, MAX_WEIGHT)
        w /= w.sum()
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        return w, r, v, s
    return None, None, None, None


# ─────────────────────────────────────────────
# Today's Allocation
# Last WINDOW days → optimize → live signal
# ─────────────────────────────────────────────

def compute_todays_allocation(prices, fg_df, current_score, current_rating):
    returns    = compute_returns(prices)
    last_5     = returns.iloc[-WINDOW:]
    today_date = datetime.today().strftime("%Y-%m-%d")

    in_market = current_score > CNN_FEAR_THRESHOLD

    print(f"\nToday's Date:      {today_date}")
    print(f"CNN F&G Score:     {current_score:.1f} ({current_rating.title()})")
    print(f"Market Signal:     {'NORMAL — Investing' if in_market else 'EXTREME FEAR — Holding Cash'}")

    if not in_market:
        print("CNN Fear Gate: HOLD CASH")
        allocation = pd.DataFrame([{
            "date":       today_date,
            "ticker":     "CASH",
            "sector":     "N/A",
            "weight":     1.0,
            "cnn_score":  round(current_score, 2),
            "cnn_rating": current_rating,
            "signal":     "HOLD CASH — Extreme Fear"
        }])
        return allocation

    selected = select_top_stocks(last_5)
    weights, ann_ret, ann_vol, sharpe = optimize_portfolio(last_5[selected])

    if weights is None:
        weights = np.full(len(selected), 1.0 / len(selected))
        ann_ret = ann_vol = sharpe = None
        print("  Optimizer fell back to equal weights")

    rows = []
    print(f"\n{'Ticker':<8} {'Sector':<14} {'Weight':>8}")
    print("-" * 32)
    for ticker, w in zip(selected, weights):
        print(f"{ticker:<8} {SECTOR_MAP.get(ticker, 'N/A'):<14} {w*100:>7.1f}%")
        rows.append({
            "date":       today_date,
            "ticker":     ticker,
            "sector":     SECTOR_MAP.get(ticker, "N/A"),
            "weight":     round(w, 4),
            "cnn_score":  round(current_score, 2),
            "cnn_rating": current_rating,
            "signal":     "INVEST"
        })

    if sharpe:
        print(f"\nExpected Ann. Return:  {ann_ret*100:.2f}%")
        print(f"Expected Ann. Vol:     {ann_vol*100:.2f}%")
        print(f"Expected Sharpe:       {sharpe:.4f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Strategy 1 — MPT Sliding Window Backtest
# ─────────────────────────────────────────────

def run_mpt_backtest(prices):
    returns   = compute_returns(prices)
    all_dates = returns.index

    portfolio_val     = INITIAL_CAPITAL
    portfolio_values  = []
    rebalance_records = []

    print(f"\nRunning MPT {WINDOW}-day sliding window backtest ({len(all_dates)} trading days)...")

    for i in range(WINDOW, len(all_dates)):
        window_ret = returns.iloc[i - WINDOW: i]
        today_date = all_dates[i]
        today_ret  = returns.iloc[i]

        selected = select_top_stocks(window_ret)
        if len(selected) < 2:
            portfolio_values.append((today_date, portfolio_val))
            continue

        weights, ann_ret, ann_vol, sharpe = optimize_portfolio(window_ret[selected])

        if weights is None:
            weights = np.full(len(selected), 1.0 / len(selected))
            ann_ret = ann_vol = sharpe = None

        daily_return  = np.dot(weights, today_ret[selected].values)
        portfolio_val *= (1 + daily_return)

        portfolio_values.append((today_date, portfolio_val))
        rebalance_records.append({
            "date":       today_date,
            "stocks":     selected,
            "weights":    dict(zip(selected, weights.round(4))),
            "ann_return": round(ann_ret, 4) if ann_ret else None,
            "ann_vol":    round(ann_vol, 4) if ann_vol else None,
            "sharpe":     round(sharpe, 4)  if sharpe  else None
        })

    port_df = pd.DataFrame(portfolio_values, columns=["Date", "Value"]).set_index("Date")
    return port_df, rebalance_records


# ─────────────────────────────────────────────
# Strategy 2 — Moving Average Crossover
# 5-day MA > 20-day MA → invested in SPY
# 5-day MA < 20-day MA → hold cash
# ─────────────────────────────────────────────

def run_ma_strategy(spy_series):
    print("Running Moving Average Crossover strategy...")
    df = pd.DataFrame({"price": spy_series})
    df["ma_short"] = df["price"].rolling(MA_SHORT).mean()
    df["ma_long"]  = df["price"].rolling(MA_LONG).mean()
    df = df.dropna()

    val    = INITIAL_CAPITAL
    values = []
    for i in range(1, len(df)):
        today     = df.index[i]
        ret       = df["price"].iloc[i] / df["price"].iloc[i - 1] - 1
        in_market = df["ma_short"].iloc[i - 1] > df["ma_long"].iloc[i - 1]
        if in_market:
            val *= (1 + ret)
        values.append((today, val))

    ma_df = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")
    return ma_df


# ─────────────────────────────────────────────
# Strategy 3 — CNN Fear & Greed Gated MPT
#
# Runs MPT normally when CNN score > 25
# Holds cash when CNN score <= 25 (Extreme Fear)
# ─────────────────────────────────────────────

def run_cnn_strategy(prices, fg_df):
    print("Running CNN Fear & Greed gated MPT strategy...")
    returns   = compute_returns(prices)
    all_dates = returns.index

    portfolio_val    = INITIAL_CAPITAL
    portfolio_values = []
    days_in_cash     = 0
    days_invested    = 0

    for i in range(WINDOW, len(all_dates)):
        window_ret = returns.iloc[i - WINDOW: i]
        today_date = all_dates[i]
        today_ret  = returns.iloc[i]

        # Use previous day's CNN score to avoid lookahead bias
        prev_date = all_dates[i - 1]
        if prev_date in fg_df.index:
            prev_score = fg_df.loc[prev_date, "score"]
        else:
            # Find nearest available date
            available = fg_df.index[fg_df.index <= prev_date]
            prev_score = fg_df.loc[available[-1], "score"] if len(available) > 0 else 50.0

        # Fear gate
        if prev_score <= CNN_FEAR_THRESHOLD:
            days_in_cash += 1
            portfolio_values.append((today_date, portfolio_val))
            continue

        selected = select_top_stocks(window_ret)
        if len(selected) < 2:
            portfolio_values.append((today_date, portfolio_val))
            continue

        weights, _, _, _ = optimize_portfolio(window_ret[selected])
        if weights is None:
            weights = np.full(len(selected), 1.0 / len(selected))

        daily_return  = np.dot(weights, today_ret[selected].values)
        portfolio_val *= (1 + daily_return)
        days_invested += 1
        portfolio_values.append((today_date, portfolio_val))

    total_days = days_in_cash + days_invested
    pct_cash   = days_in_cash / total_days * 100 if total_days > 0 else 0
    print(f"  CNN strategy: {days_invested} days invested, "
          f"{days_in_cash} days in cash ({pct_cash:.1f}% cash rate)")

    cnn_df = pd.DataFrame(portfolio_values, columns=["Date", "Value"]).set_index("Date")
    return cnn_df


# ─────────────────────────────────────────────
# S&P 500 Buy-and-Hold Baseline
# ─────────────────────────────────────────────

def build_spy_baseline(spy_series, port_index):
    spy_ret = spy_series.pct_change().dropna()
    val     = INITIAL_CAPITAL
    values  = []
    for date, r in spy_ret.items():
        val *= (1 + r)
        values.append((date, val))

    spy_df = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")
    port_index_norm = pd.to_datetime(port_index).normalize()
    common = port_index_norm.intersection(spy_df.index)

    if common.empty:
        raise ValueError(
            f"No date overlap between SPY and portfolio.\n"
            f"SPY:  {spy_df.index[0]} → {spy_df.index[-1]}\n"
            f"Port: {port_index[0]} → {port_index[-1]}"
        )

    spy_df = spy_df.loc[common]
    spy_df["Value"] = spy_df["Value"] / spy_df["Value"].iloc[0] * INITIAL_CAPITAL
    return spy_df


# ─────────────────────────────────────────────
# Align all strategies to common dates
# ─────────────────────────────────────────────

def align_strategies(port_df, spy_df, ma_df, cnn_df):
    common = (port_df.index
              .intersection(spy_df.index)
              .intersection(ma_df.index)
              .intersection(cnn_df.index))

    port_df = port_df.loc[common].copy()
    spy_df  = spy_df.loc[common].copy()
    ma_df   = ma_df.loc[common].copy()
    cnn_df  = cnn_df.loc[common].copy()

    for df in [spy_df, ma_df, cnn_df]:
        df["Value"] = df["Value"] / df["Value"].iloc[0] * INITIAL_CAPITAL

    return port_df, spy_df, ma_df, cnn_df


# ─────────────────────────────────────────────
# Compute Summary Metrics
# ─────────────────────────────────────────────

def compute_metrics(series, label):
    daily        = series.pct_change().dropna()
    total_return = (series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sharpe       = (daily.mean() - RISK_FREE_RATE) / daily.std() * np.sqrt(252)
    max_drawdown = ((series / series.cummax()) - 1).min() * 100
    return {
        "Strategy":           label,
        "Final Value ($)":    round(series.iloc[-1], 2),
        "Total Return (%)":   round(total_return, 2),
        "Annualized Sharpe":  round(sharpe, 4),
        "Max Drawdown (%)":   round(max_drawdown, 2)
    }


# ─────────────────────────────────────────────
# Plot 1 — Efficient Frontier
# ─────────────────────────────────────────────

def plot_efficient_frontier(returns, selected_stocks, optimal_weights):
    print("Plotting efficient frontier...")
    mean_ret = returns[selected_stocks].mean()
    cov_mat  = returns[selected_stocks].cov()
    n        = len(selected_stocks)

    sim_r, sim_v, sim_s = [], [], []
    for _ in range(3000):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        sim_r.append(r); sim_v.append(v); sim_s.append(s)

    opt_r, opt_v, opt_s = portfolio_performance(optimal_weights, mean_ret, cov_mat)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(sim_v, sim_r, c=sim_s, cmap="viridis", alpha=0.4, s=12,
                    label="Random Portfolios")
    ax.scatter(opt_v, opt_r, color="red", s=200, zorder=5, marker="*",
               label=f"Max Sharpe = {opt_s:.2f}")
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Return",     fontsize=12)
    ax.set_title(f"Efficient Frontier — {START_DATE} to {END_DATE}", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = OUTPUT_DIR + "efficient_frontier.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Plot 2 — Portfolio Value: All 4 Strategies
# ─────────────────────────────────────────────

def plot_performance(port_df, spy_df, ma_df, cnn_df):
    print("Plotting performance comparison...")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(port_df.index, port_df["Value"], label="MPT Portfolio",
            color="steelblue",  linewidth=2)
    ax.plot(spy_df.index,  spy_df["Value"],  label="S&P 500 (Buy & Hold)",
            color="darkorange", linewidth=2, linestyle="--")
    ax.plot(ma_df.index,   ma_df["Value"],   label="MA Crossover (5/20)",
            color="green",      linewidth=2, linestyle="-.")
    ax.plot(cnn_df.index,  cnn_df["Value"],  label="CNN Fear Gated MPT",
            color="crimson",    linewidth=2, linestyle=":")
    ax.set_xlabel("Date",                fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title(f"Strategy Comparison — $10,000 Starting Capital "
                 f"({START_DATE} to {END_DATE})", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "performance_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Plot 3 — Daily Allocation Shift
# ─────────────────────────────────────────────

def plot_allocation_shift(rebalance_records):
    print("Plotting allocation shift...")

    # Sample every 5 days
    sampled = rebalance_records[::5]
    dates   = [r["date"] for r in sampled]

    # Group weights by sector
    sector_weights = []
    for r in sampled:
        row = {s: 0.0 for s in TICKERS.keys()}
        for ticker, w in r["weights"].items():
            sector = SECTOR_MAP.get(ticker, None)
            if sector:
                row[sector] += w
        sector_weights.append(row)

    df_s = pd.DataFrame(sector_weights, index=dates)

    sector_colors = {
        "Tech":        "#58a6ff",
        "Healthcare":  "#3fb950",
        "Utilities":   "#e3b341",
        "Commodities": "#f0883e",
        "Finance":     "#bc8cff"
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom  = np.zeros(len(df_s))

    for sector in TICKERS.keys():
        vals = df_s[sector].values
        ax.bar(df_s.index, vals, bottom=bottom,
               label=sector, color=sector_colors[sector],
               width=3, alpha=0.88)
        bottom += vals

    ax.set_xlabel("Date",   fontsize=11)
    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title("Sector Allocation Shift Over Time (MPT)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10, ncol=1)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "allocation_shift.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Plot 4 — Rolling 21-Day Sharpe Ratio
# ─────────────────────────────────────────────

def plot_rolling_sharpe(port_df, spy_df, ma_df, cnn_df):
    print("Plotting rolling Sharpe...")
    roll = 21

    def rolling_sharpe(series):
        daily = series.pct_change().dropna()
        return (daily.rolling(roll).mean() - RISK_FREE_RATE) / \
                daily.rolling(roll).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(rolling_sharpe(port_df["Value"]).index,
            rolling_sharpe(port_df["Value"]),
            label="MPT Portfolio",        color="steelblue",  linewidth=1.5)
    ax.plot(rolling_sharpe(spy_df["Value"]).index,
            rolling_sharpe(spy_df["Value"]),
            label="S&P 500 (Buy & Hold)", color="darkorange", linewidth=1.5, linestyle="--")
    ax.plot(rolling_sharpe(ma_df["Value"]).index,
            rolling_sharpe(ma_df["Value"]),
            label="MA Crossover (5/20)",  color="green",      linewidth=1.5, linestyle="-.")
    ax.plot(rolling_sharpe(cnn_df["Value"]).index,
            rolling_sharpe(cnn_df["Value"]),
            label="CNN Fear Gated MPT",   color="crimson",    linewidth=1.5, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date",                       fontsize=11)
    ax.set_ylabel(f"Rolling {roll}-Day Sharpe", fontsize=11)
    ax.set_title(f"Rolling {roll}-Day Sharpe Ratio — All Strategies", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "rolling_sharpe.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Plot 5 — CNN Fear & Greed Index Over Time
# ─────────────────────────────────────────────

def plot_cnn_fear_greed(fg_df):
    print("Plotting CNN Fear & Greed Index...")

    scores = fg_df["score"]

    fig, ax = plt.subplots(figsize=(13, 4))

    # Color zones
    ax.axhspan(0,  25,  alpha=0.08, color="red",    label="Extreme Fear (0–25)")
    ax.axhspan(25, 45,  alpha=0.06, color="orange",  label="Fear (25–45)")
    ax.axhspan(45, 55,  alpha=0.05, color="yellow",  label="Neutral (45–55)")
    ax.axhspan(55, 75,  alpha=0.06, color="lightgreen", label="Greed (55–75)")
    ax.axhspan(75, 100, alpha=0.08, color="green",   label="Extreme Greed (75–100)")

    ax.plot(scores.index, scores.values,
            color="black", linewidth=1.5, label="CNN F&G Score", zorder=5)

    ax.axhline(CNN_FEAR_THRESHOLD, color="red", linewidth=1.2,
               linestyle="--", label=f"Cash Threshold ({CNN_FEAR_THRESHOLD})")

    # Shade cash zones
    ax.fill_between(scores.index, 0, scores.values,
                    where=(scores.values <= CNN_FEAR_THRESHOLD),
                    alpha=0.25, color="red", label="Portfolio Holds Cash")

    ax.set_ylim(0, 100)
    ax.set_xlabel("Date",                    fontsize=11)
    ax.set_ylabel("Fear & Greed Score",      fontsize=11)
    ax.set_title("CNN Fear & Greed Index — Red Zones = Portfolio Holds Cash", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = OUTPUT_DIR + "cnn_fear_greed.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Print + Save Summary
# ─────────────────────────────────────────────

def print_and_save_summary(port_df, spy_df, ma_df, cnn_df,
                            rebalance_records, elapsed):
    metrics = [
        compute_metrics(port_df["Value"], "MPT Portfolio"),
        compute_metrics(spy_df["Value"],  "S&P 500 Buy & Hold"),
        compute_metrics(ma_df["Value"],   "MA Crossover (5/20)"),
        compute_metrics(cnn_df["Value"],  "CNN Fear Gated MPT"),
    ]

    print("\n" + "=" * 74)
    print("PERFORMANCE SUMMARY")
    print("=" * 74)
    print(f"{'Metric':<26} {'MPT':>11} {'S&P 500':>11} {'MA Cross':>11} {'CNN MPT':>11}")
    print("-" * 74)
    for key in ["Final Value ($)", "Total Return (%)", "Annualized Sharpe", "Max Drawdown (%)"]:
        vals = [str(m[key]) for m in metrics]
        print(f"{key:<26} {vals[0]:>11} {vals[1]:>11} {vals[2]:>11} {vals[3]:>11}")
    print("=" * 74)
    print(f"\nTotal runtime: {elapsed:.1f} seconds")

    summary_df = pd.DataFrame(metrics)
    path = OUTPUT_DIR + "summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data ---
    prices             = download_prices()
    spy                = download_spy()
    fg_df, cur_score, cur_rating = download_cnn_fear_greed()

    # --- Today's live allocation ---
    print("\n" + "=" * 50)
    print("TODAY'S RECOMMENDED ALLOCATION")
    print("=" * 50)
    todays_allocation = compute_todays_allocation(
        prices, fg_df, cur_score, cur_rating)
    todays_allocation.to_csv(OUTPUT_DIR + "todays_allocation.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}todays_allocation.csv")

    # --- Full period backtest ---
    returns          = compute_returns(prices)
    selected_initial = select_top_stocks(returns)
    print(f"\nInitial stock selection (full period): {selected_initial}")

    opt_w, _, _, opt_s = optimize_portfolio(returns[selected_initial])
    if opt_w is None:
        opt_w = np.full(len(selected_initial), 1.0 / len(selected_initial))
    if opt_s:
        print(f"Optimal Sharpe (full period): {opt_s:.4f}")

    plot_efficient_frontier(returns, selected_initial, opt_w)

    # --- Run all strategies ---
    port_df, rebalance_records = run_mpt_backtest(prices)
    spy_df                     = build_spy_baseline(spy, port_df.index)
    ma_df                      = run_ma_strategy(spy)
    cnn_df                     = run_cnn_strategy(prices, fg_df)

    # --- Align to common dates ---
    port_df, spy_df, ma_df, cnn_df = align_strategies(
        port_df, spy_df, ma_df, cnn_df)

    # --- Generate all 5 plots ---
    plot_performance(port_df, spy_df, ma_df, cnn_df)
    plot_allocation_shift(rebalance_records)
    plot_rolling_sharpe(port_df, spy_df, ma_df, cnn_df)
    plot_cnn_fear_greed(fg_df)

    # --- Save portfolio values ---
    port_df.to_csv(OUTPUT_DIR + "portfolio_values.csv")
    print(f"Saved: {OUTPUT_DIR}portfolio_values.csv")

    # --- Final summary ---
    elapsed = time.time() - START_TIME
    print_and_save_summary(port_df, spy_df, ma_df, cnn_df,
                           rebalance_records, elapsed)


if __name__ == "__main__":
    main()
