from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="S&P 500 섹터 대장주 RSI/MACD 모니터", layout="wide")
st.title("미국 S&P 500 섹터별 대장주 실시간 RSI/MACD 모니터")
st.caption("섹터 대장주는 시가총액 기준으로 주 1회 갱신됩니다.")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=True)
def get_sector_leaders_weekly() -> pd.DataFrame:
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(wiki_url, headers=headers)
    sp500 = pd.read_html(response.text)[0]
    sp500 = sp500.rename(columns={"Symbol": "Ticker", "GICS Sector": "Sector"})
    sp500["Ticker"] = sp500["Ticker"].str.replace(".", "-", regex=False)

    rows = []
    for _, row in sp500[["Ticker", "Sector"]].iterrows():
        ticker = row["Ticker"]
        sector = row["Sector"]
        market_cap = None
        try:
            info = yf.Ticker(ticker).fast_info
            market_cap = info.get("marketCap")
        except Exception:
            market_cap = None
        rows.append({"Sector": sector, "Ticker": ticker, "MarketCap": market_cap})

    cap_df = pd.DataFrame(rows).dropna(subset=["MarketCap"])
    leader_df = cap_df.sort_values("MarketCap", ascending=False).groupby("Sector", as_index=False).first()
    return leader_df.sort_values("Sector").reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_hourly_with_indicators(ticker: str, period: str = "60d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=False)
    if data is None or data.empty:
        return pd.DataFrame()

    df = flatten_columns(data.copy()).reset_index()
    if "Datetime" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    if "Datetime" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Datetime"})

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["RSI14"] = compute_rsi(df["Close"], window=14)
    macd, signal, hist = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    return df


def crossed_up(series_a: pd.Series, series_b: pd.Series) -> bool:
    if len(series_a) < 2 or len(series_b) < 2:
        return False
    prev_a, curr_a = series_a.iloc[-2], series_a.iloc[-1]
    prev_b, curr_b = series_b.iloc[-2], series_b.iloc[-1]
    if pd.isna(prev_a) or pd.isna(curr_a) or pd.isna(prev_b) or pd.isna(curr_b):
        return False
    return prev_a <= prev_b and curr_a > curr_b


def crossed_down(series_a: pd.Series, series_b: pd.Series) -> bool:
    if len(series_a) < 2 or len(series_b) < 2:
        return False
    prev_a, curr_a = series_a.iloc[-2], series_a.iloc[-1]
    prev_b, curr_b = series_b.iloc[-2], series_b.iloc[-1]
    if pd.isna(prev_a) or pd.isna(curr_a) or pd.isna(prev_b) or pd.isna(curr_b):
        return False
    return prev_a >= prev_b and curr_a < curr_b


def rsi_oversold_exit(rsi: pd.Series, threshold: float = 30.0) -> bool:
    if len(rsi) < 2:
        return False
    prev_rsi, curr_rsi = rsi.iloc[-2], rsi.iloc[-1]
    if pd.isna(prev_rsi) or pd.isna(curr_rsi):
        return False
    return prev_rsi <= threshold and curr_rsi > threshold


if st.button("지금 새로고침"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("섹터별 대장주를 불러오는 중..."):
    leaders = get_sector_leaders_weekly()

if leaders.empty:
    st.error("섹터 대장주 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

summary_rows = []
chart_payload = {}

for _, row in leaders.iterrows():
    sector = row["Sector"]
    ticker = row["Ticker"]
    hist = get_hourly_with_indicators(ticker)
    if hist.empty:
        summary_rows.append(
            {
                "섹터명": sector,
                "현재 대장주 티커": ticker,
                "현재가": None,
                "매수 신호 상태": "데이터 없음",
                "매도 신호 상태": "데이터 없음",
            }
        )
        continue

    latest = hist.iloc[-1]
    buy_signal = rsi_oversold_exit(hist["RSI14"]) and crossed_up(hist["MACD"], hist["MACD_Signal"])
    sell_signal = (float(latest["RSI14"]) > 80.0) and crossed_down(hist["MACD"], hist["MACD_Signal"])

    summary_rows.append(
        {
            "섹터명": sector,
            "현재 대장주 티커": ticker,
            "현재가": round(float(latest["Close"]), 2),
            "매수 신호 상태": "ON" if buy_signal else "OFF",
            "매도 신호 상태": "ON" if sell_signal else "OFF",
        }
    )
    chart_payload[ticker] = {"sector": sector, "df": hist}

summary_df = pd.DataFrame(summary_rows).sort_values("섹터명").reset_index(drop=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("섹터별 대장주 신호 현황")
with col2:
    st.caption(f"업데이트 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("종목별 1시간봉 차트 (가격/RSI/MACD)")

for ticker, payload in chart_payload.items():
    sector = payload["sector"]
    df = payload["df"]

    with st.expander(f"{sector} - {ticker}", expanded=False):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(df["Datetime"], df["Close"], color="#1f77b4", linewidth=1.3, label="Close")
        axes[0].set_title(f"{ticker} Price (1h)")
        axes[0].grid(alpha=0.3, linestyle="--")
        axes[0].legend(loc="upper left")

        axes[1].plot(df["Datetime"], df["RSI14"], color="#ff7f0e", linewidth=1.2, label="RSI14")
        axes[1].axhline(30, color="green", linestyle="--", linewidth=1.0)
        axes[1].axhline(70, color="red", linestyle="--", linewidth=1.0)
        axes[1].axhline(80, color="purple", linestyle=":", linewidth=1.0)
        axes[1].set_ylim(0, 100)
        axes[1].set_title("RSI")
        axes[1].grid(alpha=0.3, linestyle="--")
        axes[1].legend(loc="upper left")

        axes[2].plot(df["Datetime"], df["MACD"], color="#2ca02c", linewidth=1.2, label="MACD")
        axes[2].plot(df["Datetime"], df["MACD_Signal"], color="#d62728", linewidth=1.2, label="Signal")
        axes[2].bar(df["Datetime"], df["MACD_Hist"], color="#7f7f7f", alpha=0.4, width=0.03, label="Hist")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title("MACD")
        axes[2].grid(alpha=0.3, linestyle="--")
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.caption("참고: 실시간 데이터 지연 또는 누락은 Yahoo Finance 원천 데이터 상태에 따라 발생할 수 있습니다.")