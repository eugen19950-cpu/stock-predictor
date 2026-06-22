
import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="종목 예측기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 종목 예측기 V3 Web")
st.caption("종목명 검색 + 예상가격 + 구간별 확률 + 목표주가 도달확률")

st.warning(
    "이 결과는 과거 변동성과 평균수익률 기반 몬테카를로 시뮬레이션입니다. "
    "뉴스, 실적, 수급, 공시, 이벤트는 반영하지 않습니다."
)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_krx():
    krx = fdr.StockListing("KRX")

    krx = krx[
        krx["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])
    ].copy()

    rows = []

    for _, row in krx.iterrows():
        name = row["Name"]
        code = str(row["Code"]).zfill(6)
        market = row["Market"]

        suffix = ".KS" if market == "KOSPI" else ".KQ"
        yahoo_ticker = code + suffix

        rows.append({
            "종목명": name,
            "코드": code,
            "시장": "KOSPI" if market == "KOSPI" else "KOSDAQ",
            "티커": yahoo_ticker
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price(ticker):
    return yf.download(
        ticker,
        period="2y",
        auto_adjust=True,
        progress=False
    )


def simulate_prices(close, current_price, days=20, sims=5000, seed=42):
    returns = close.pct_change().dropna()

    mu = float(returns.mean())
    sigma = float(returns.std())

    rng = np.random.default_rng(seed)

    prices = []

    for _ in range(sims):
        price = current_price

        for _ in range(days):
            r = rng.normal(mu, sigma)
            price *= (1 + r)

        prices.append(price)

    return np.array(prices)


def calc_forecast_stats(prices):
    return {
        "평균 예상가": np.mean(prices),
        "약세 10% 구간": np.percentile(prices, 10),
        "보수 25% 구간": np.percentile(prices, 25),
        "중립 50% 구간": np.percentile(prices, 50),
        "강세 75% 구간": np.percentile(prices, 75),
        "매우강세 90% 구간": np.percentile(prices, 90),
        "시뮬레이션 최저": np.min(prices),
        "시뮬레이션 최고": np.max(prices),
    }


def calc_band_probs(prices, current_price):
    returns = (prices / current_price - 1) * 100

    bands = [
        ("하락", -9999, 0),
        ("0~+10%", 0, 10),
        ("+10~+20%", 10, 20),
        ("+20~+30%", 20, 30),
        ("+30~+50%", 30, 50),
        ("+50% 이상", 50, 9999),
    ]

    result = []

    for name, low, high in bands:
        prob = np.mean((returns >= low) & (returns < high)) * 100
        result.append({
            "구간": name,
            "확률": f"{prob:.1f}%"
        })

    return pd.DataFrame(result)


def calc_target_probs(prices, current_price):
    target_rates = [10, 20, 30, 50, 100]

    result = []

    for rate in target_rates:
        target_price = current_price * (1 + rate / 100)
        prob = np.mean(prices >= target_price) * 100

        result.append({
            "목표상승률": f"+{rate}%",
            "목표가": f"{target_price:,.0f}원",
            "도달확률": f"{prob:.1f}%"
        })

    return pd.DataFrame(result)


def money(x):
    return f"{float(x):,.0f}원"


with st.spinner("📡 KRX 종목 목록 불러오는 중입니다... 처음 실행 시 시간이 걸릴 수 있어요."):
    krx_df = load_krx()

st.success(f"✅ 종목 목록 로딩 완료: {len(krx_df):,}개 종목")

with st.sidebar:
    st.header("설정")

    query = st.text_input(
        "종목명 검색",
        value="리노공업",
        placeholder="예: 리노공업, 티엘비, 삼성전자"
    )

    sims = st.slider(
        "시뮬레이션 횟수",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000
    )

    custom_target_text = st.text_input(
        "사용자 목표주가",
        value="",
        placeholder="예: 150000"
    )

    seed = st.number_input(
        "랜덤 시드",
        value=42,
        step=1
    )

    run = st.button("예측 실행", type="primary")


if not query:
    st.info("왼쪽에서 종목명을 입력해 주세요.")
    st.stop()


candidates = krx_df[
    krx_df["종목명"].str.contains(query, case=False, na=False)
]

if len(candidates) == 0:
    st.error("검색된 종목이 없습니다. 종목명을 다시 입력해 주세요.")
    st.stop()

selected_label = st.selectbox(
    "종목 선택",
    candidates.apply(
        lambda r: f"{r['종목명']} | {r['시장']} | {r['코드']}",
        axis=1
    ).tolist()
)

selected_code = selected_label.split("|")[-1].strip()
selected_row = candidates[candidates["코드"] == selected_code].iloc[0]

st.subheader(f"선택 종목: {selected_row['종목명']}")
st.write(
    f"시장: **{selected_row['시장']}** / "
    f"코드: **{selected_row['코드']}** / "
    f"야후티커: **{selected_row['티커']}**"
)

custom_target = None

if custom_target_text.strip():
    try:
        custom_target = float(custom_target_text.replace(",", ""))
    except Exception:
        st.error("사용자 목표주가는 숫자로 입력해 주세요.")
        st.stop()


if run:
    with st.spinner("📈 주가 데이터 다운로드 및 시뮬레이션 중입니다..."):
        df = download_price(selected_row["티커"])

        if df is None or len(df) < 200:
            st.error("데이터가 부족하여 분석할 수 없습니다.")
            st.stop()

        close = df["Close"].dropna()

        if len(close) < 200:
            st.error("종가 데이터가 부족합니다.")
            st.stop()

        current_price = float(close.iloc[-1])

        st.metric("현재가", money(current_price))

        tabs = st.tabs(["20거래일", "60거래일", "90거래일"])

        for tab, days in zip(tabs, [20, 60, 90]):
            with tab:
                prices = simulate_prices(
                    close=close,
                    current_price=current_price,
                    days=days,
                    sims=sims,
                    seed=int(seed) + days
                )

                stats = calc_forecast_stats(prices)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("중립 50% 예상가", money(stats["중립 50% 구간"]))
                c2.metric("강세 75% 예상가", money(stats["강세 75% 구간"]))
                c3.metric("매우강세 90% 예상가", money(stats["매우강세 90% 구간"]))
                c4.metric("평균 예상가", money(stats["평균 예상가"]))

                st.write("### 📊 예상가격")
                price_df = pd.DataFrame([
                    {
                        "구분": key,
                        "가격": money(value),
                        "현재가 대비": f"{(value / current_price - 1) * 100:+.1f}%"
                    }
                    for key, value in stats.items()
                ])

                st.dataframe(price_df, use_container_width=True)

                st.write("### 📌 구간별 확률")
                st.dataframe(
                    calc_band_probs(prices, current_price),
                    use_container_width=True
                )

                st.write("### 🎯 목표 상승률별 도달확률")
                st.dataframe(
                    calc_target_probs(prices, current_price),
                    use_container_width=True
                )

                if custom_target is not None:
                    custom_prob = np.mean(prices >= custom_target) * 100
                    need_return = (custom_target / current_price - 1) * 100

                    st.write("### 💰 사용자 목표주가")
                    st.success(
                        f"목표가 {custom_target:,.0f}원 "
                        f"({need_return:+.1f}%)의 "
                        f"{days}거래일 도달확률: {custom_prob:.1f}%"
                    )
else:
    st.info("왼쪽 설정에서 종목을 고른 뒤 **예측 실행** 버튼을 눌러주세요.")
