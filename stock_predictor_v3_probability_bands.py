
# stock_predictor_v3_probability_bands.py
# 독립 실행형 종목 예측기 V3
# - 종목명 검색 가능
# - 20일 / 60일 / 90일 예상가격 출력
# - 각 가격 구간별 확률표 출력
# - 목표주가 직접 입력 시 도달확률 계산

import subprocess
import sys

for pkg in ["finance-datareader", "yfinance"]:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

import FinanceDataReader as fdr
import yfinance as yf
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print("=" * 100)
print("📈 종목 예측기 V3.0")
print("종목명 검색 + 예상가격 + 구간별 확률 + 목표주가 도달확률")
print("=" * 100)

print("\nKRX 종목 목록 불러오는 중...")

krx = fdr.StockListing("KRX")

krx = krx[
    krx["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])
].copy()

name_to_info = {}

for _, row in krx.iterrows():
    name = row["Name"]
    code = str(row["Code"]).zfill(6)
    market = row["Market"]

    suffix = ".KS" if market == "KOSPI" else ".KQ"
    yahoo_ticker = code + suffix

    name_to_info[name] = {
        "code": code,
        "market": "KOSPI" if market == "KOSPI" else "KOSDAQ",
        "ticker": yahoo_ticker
    }

print(f"검색 가능 종목 수: {len(name_to_info)}개")


def find_stock_name(query):
    query = query.strip()

    if query in name_to_info:
        return query

    candidates = [
        name for name in name_to_info.keys()
        if query in name
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    print("\n비슷한 종목이 여러 개 있습니다.")
    for i, name in enumerate(candidates[:20], 1):
        info = name_to_info[name]
        print(f"{i:2d}. {name} ({info['market']}, {info['code']})")

    try:
        choice = input("번호 선택: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(candidates[:20]):
            return candidates[idx]
    except Exception:
        pass

    return None


def simulate_prices(close, current_price, days=20, sims=5000):
    returns = close.pct_change().dropna()

    mu = float(returns.mean())
    sigma = float(returns.std())

    prices = []

    for _ in range(sims):
        price = current_price

        for _ in range(days):
            r = np.random.normal(mu, sigma)
            price *= (1 + r)

        prices.append(price)

    return np.array(prices)


def calc_forecast_stats(prices):
    return {
        "p10": np.percentile(prices, 10),
        "p25": np.percentile(prices, 25),
        "p50": np.percentile(prices, 50),
        "p75": np.percentile(prices, 75),
        "p90": np.percentile(prices, 90),
        "min": np.min(prices),
        "max": np.max(prices),
        "mean": np.mean(prices),
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
        result.append((name, prob))

    return result


def calc_target_probs(prices, current_price):
    target_rates = [10, 20, 30, 50, 100]
    result = []

    for rate in target_rates:
        target_price = current_price * (1 + rate / 100)
        prob = np.mean(prices >= target_price) * 100
        result.append((rate, target_price, prob))

    return result


def print_forecast(days, prices, current_price, custom_target=None):
    stats = calc_forecast_stats(prices)

    print("\n" + "=" * 100)
    print(f"📊 {days}거래일 예상가격")
    print("=" * 100)

    print(f"현재가              : {current_price:,.0f}원")
    print(f"평균 예상가         : {stats['mean']:,.0f}원")
    print(f"약세 10% 구간       : {stats['p10']:,.0f}원")
    print(f"보수 25% 구간       : {stats['p25']:,.0f}원")
    print(f"중립 50% 구간       : {stats['p50']:,.0f}원")
    print(f"강세 75% 구간       : {stats['p75']:,.0f}원")
    print(f"매우강세 90% 구간   : {stats['p90']:,.0f}원")
    print(f"시뮬레이션 예상범위 : {stats['min']:,.0f}원 ~ {stats['max']:,.0f}원")

    print("\n📌 구간별 확률")
    print("-" * 100)

    for band_name, prob in calc_band_probs(prices, current_price):
        print(f"{band_name:<12} : {prob:>5.1f}%")

    print("\n🎯 목표 상승률별 도달확률")
    print("-" * 100)

    for rate, target_price, prob in calc_target_probs(prices, current_price):
        print(f"+{rate:<3d}% 목표가 {target_price:>12,.0f}원 : {prob:>5.1f}%")

    if custom_target is not None:
        custom_prob = np.mean(prices >= custom_target) * 100
        need_return = (custom_target / current_price - 1) * 100

        print("\n💰 사용자 목표주가")
        print("-" * 100)
        print(f"사용자 목표가       : {custom_target:,.0f}원")
        print(f"필요 상승률         : {need_return:+.1f}%")
        print(f"{days}거래일 도달확률 : {custom_prob:.1f}%")


while True:
    query = input("\n종목명 입력 (종료=끝): ").strip()

    if query == "끝":
        print("종목 예측기를 종료합니다.")
        break

    selected_name = find_stock_name(query)

    if selected_name is None:
        print("종목을 찾지 못했습니다. 예: 리노공업, 티엘비, 삼성전자")
        continue

    info = name_to_info[selected_name]

    print("\n" + "=" * 100)
    print(f"선택 종목 : {selected_name}")
    print(f"시장      : {info['market']}")
    print(f"코드      : {info['code']}")
    print(f"야후티커  : {info['ticker']}")
    print("=" * 100)

    target_input = input("사용자 목표주가 입력 (엔터시 생략): ").strip()

    custom_target = None

    if target_input:
        try:
            custom_target = float(target_input.replace(",", ""))
        except Exception:
            print("목표주가 입력 오류. 사용자 목표주가는 생략합니다.")
            custom_target = None

    print("주가 데이터 다운로드 중...")

    df = yf.download(
        info["ticker"],
        period="2y",
        auto_adjust=True,
        progress=False
    )

    if df is None or len(df) < 200:
        print("데이터가 부족하여 분석할 수 없습니다.")
        continue

    close = df["Close"].dropna()

    if len(close) < 200:
        print("종가 데이터가 부족합니다.")
        continue

    current_price = float(close.iloc[-1])

    for days in [20, 60, 90]:
        prices = simulate_prices(
            close=close,
            current_price=current_price,
            days=days,
            sims=5000
        )

        print_forecast(
            days=days,
            prices=prices,
            current_price=current_price,
            custom_target=custom_target
        )

    print("\n참고: 이 결과는 과거 변동성과 평균수익률 기반 몬테카를로 시뮬레이션입니다.")
    print("뉴스, 실적, 수급, 이벤트는 반영하지 않습니다.")
