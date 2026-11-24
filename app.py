import streamlit as st
import pandas as pd
import datetime
import time

# 1. 페이지 설정 및 다크모드 무력화 CSS
st.set_page_config(page_title="HUFS AI 통합 플래너", layout="wide")

st.markdown("""
    <style>
    /* 배경색과 글자색을 시스템 설정보다 우선하여 강제 적용 */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    
   /* 모든 텍스트를 진한 검정색으로 고정 */
    h1, h2, h3, h4, p, span, div, label {
        color: #1D1D1F !important;
    }

    /* 사이드바 가독성 확보 */
    [data-testid="stSidebar"] {
        background-color: #F2F2F7 !important;
        border-right: 1px solid #D1D1D6 !important;
    }

    /* 달력 카드: 다크모드에서도 무조건 흰색 배경 유지 */
    .day-box {
        border: 1px solid #D1D1D6 !important;
        padding: 12px;
        border-radius: 12px;
        background-color: #FFFFFF !important; /* 무조건 흰색 */
        min-height: 140px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
    }
    
    .date-head { font-weight: 800; font-size: 0.95rem; color: #3A3A3C !important; }

    /* 일정 태그: 선명한 시스템 컬러 사용 */
    .task-exam { 
        color: #FFFFFF !important; 
        font-weight: bold; 
        background-color: #FF3B30 !important; /* 시스템 레드 */
        padding: 6px; 
        border-radius: 6px; 
        font-size: 0.75rem; 
        margin-top: 5px; 
        text-align: center; 
    }
    .task-rev { 
        color: #FFFFFF !important; 
        font-weight: bold; 
        background-color: #007AFF !important; /* 시스템 블루 */
        padding: 6px; 
        border-radius: 6px; 
        font-size: 0.75rem; 
        margin-top: 4px; 
        text-align: center; 
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🗓️ HUFS AI 통합 복습 플래너")
st.write("에빙하우스 망각 곡선 기반 통합 일정 관리")

# 2. 세션 상태 관리
if 'subjects' not in st.session_state:
    st.session_state.subjects = []

# 3. 사이드바
with st.sidebar:
    st.header("📋 일정 등록")
    if st.button("➕ 과목 추가"):
        st.session_state.subjects.append({"name": f"과목 {len(st.session_state.subjects)+1}", "date": datetime.date.today()})

    for i, sub in enumerate(st.session_state.subjects):
        with st.expander(f"📚 {sub['name']}", expanded=True):
            st.session_state.subjects[i]['name'] = st.text_input("과목명", value=sub['name'], key=f"n_{i}")
            st.session_state.subjects[i]['date'] = st.date_input("시험 날짜", value=sub['date'], key=f"d_{i}")
    
    st.write("---")
    generate_btn = st.button("🪄 통합 달력 생성")

# 4. 루틴 생성 및 동적 달력 로직
if generate_btn and st.session_state.subjects:
    all_events = []
    review_intervals = [1, 3, 7, 14]
    
    # 입력된 모든 날짜 수집
    all_dates = [datetime.date.today()]
    for sub in st.session_state.subjects:
        if not sub['name']: continue
        all_events.append({"date": sub['date'], "task": f"🚩 시험: {sub['name']}", "type": "exam"})
        all_dates.append(sub['date'])
        for r in review_intervals:
            rev_date = sub['date'] - datetime.timedelta(days=r)
            all_events.append({"date": rev_date, "task": f"🔄 D-{r}: {sub['name']}", "type": "review"})
            all_dates.append(rev_date)

    # 달력 범위 계산
    start_cal = min(all_dates)
    end_cal = max(all_dates)
    
    st.success(f"✅ {start_cal}부터 {end_cal}까지의 통합 일정이 생성되었습니다.")
    
    curr = start_cal
    while curr <= end_cal:
        st.write(f"### 📅 {curr.strftime('%Y년 %m월')} 스케줄")
        cols = st.columns(7)
        for i in range(7):
            if curr > end_cal: break
            with cols[i]:
                day_tasks = [e for e in all_events if e['date'] == curr]
                is_today = " ⭐" if curr == datetime.date.today() else ""
                
                tasks_html = ""
                for t in day_tasks:
                    cls = "task-exam" if t['type'] == "exam" else "task-rev"
                    tasks_html += f"<div class='{cls}'>{t['task']}</div>"
                
                st.markdown(f"""
                <div class="day-box">
                    <div class="date-head">{curr.strftime('%m/%d')}{is_today}</div>
                    <hr style="margin:8px 0; border:0.5px solid #E5E5EA;">
                    {tasks_html}
                </div>
                """, unsafe_allow_html=True)
            curr += datetime.timedelta(days=1)
        st.write("---")
    st.balloons()

# 5. 이론 섹션 (색상 가시성 보정)
st.markdown("""
<div style="background-color: #007AFF; padding: 20px; border-radius: 15px; margin-top: 30px;">
    <h3 style="color: #FFFFFF !important; margin-top:0;">🧠 에빙하우스 망각 곡선 지능형 루틴</h3>
    <p style="color: #FFFFFF !important;">본 플래너는 뇌과학적 근거에 따라 <b>1, 3, 7, 14일 주기</b>로 복습을 배치합니다. 
    파란색은 복습일, 빨간색은 시험일입니다.</p>
</div>
""", unsafe_allow_html=True)