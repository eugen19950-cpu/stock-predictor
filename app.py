import streamlit as st

st.set_page_config(
    page_title="종목 예측기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 종목 예측기")

st.success("앱이 정상적으로 실행되었습니다.")

st.write("""
현재는 테스트 버전입니다.

여기에 앞으로:

- 종목명 검색
- 예상가격
- 구간별 확률
- 목표주가 도달확률

기능을 추가할 수 있습니다.
""")

stock_name = st.text_input(
    "종목명 입력",
    placeholder="예: 리노공업"
)

if st.button("검색"):
    if stock_name:
        st.info(f"{stock_name} 분석 기능은 다음 버전에서 연결됩니다.")
    else:
        st.warning("종목명을 입력해주세요.")
