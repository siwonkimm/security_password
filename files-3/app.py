import hashlib, math, re, time
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, requests, streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

st.set_page_config(page_title="비밀번호는 얼마나 버틸까?", page_icon="🔐", layout="wide", initial_sidebar_state="collapsed")
D = Path(__file__).parent / "data"
FONT = "Pretendard, -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, GRAY, YEL = "#5b8def", "#4cc38a", "#ff9f0a", "#ff3b30", "#af52de", "#5ac8fa", "#86868b", "#ffcc00"
KO = {"Very week": "매우 약함", "Week": "약함", "Average": "보통", "Strong": "강함", "Very strong": "매우 강함"}
ORDER = list(KO.values()); CCOL = dict(zip(ORDER, [RED, ORANGE, YEL, BLUE, GREEN]))
RCOL = {"High": RED, "Medium": ORANGE, "Low": GREEN}
YR = 31557600
TICKS = [(1, "1초"), (3600, "1시간"), (86400, "1일"), (YR, "1년"), (YR * 100, "100년"), (YR * 1e6, "100만년")]
SRC = {"t1": ("Password Metrics Dataset", "원본 100,000행 중 중복 비밀번호를 제거한 고유 25,000개(원문은 저장하지 않음). 강도 등급은 등급별 5,000개씩 균등 표본이라 실제 이용자 비율이 아님. 크랙 시간은 값 범위가 10^-7~10^104초라 로그 축·중앙값으로 집계. 'Very week' 오타는 '매우 약함'으로 정리."),
       "t2": ("Password Strength and Vulnerability Dataset", "흔한 비밀번호 500개(순위 기준). 온라인 크랙 시간은 데이터의 value·time_unit을 초로 환산(1개월=30.4일, 1년=365.25일), 오프라인은 offline_crack_sec. 카테고리별 중앙값(로그)으로 비교."),
       "t3": ("Cybersecurity Password Strength Dataset (1,000행)", "합성 데이터. 침해 위험도는 점수 구간(50 미만 High, 50~74 Medium, 75 이상 Low)과 정확히 일치해 점수에서 기계적으로 만들어진 값. 길이–점수 상관이 0.97로 길이 중심으로 생성된 것으로 보임.")}

def md(s): st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)
def human(s):
    if s is None or not np.isfinite(s): return "측정 불가"
    if s < 1e-3: return "즉시"
    if s < 1: return f"{s*1000:.0f}밀리초"
    for lim, div, u in [(60, 1, "초"), (3600, 60, "분"), (86400, 3600, "시간"), (2629800, 86400, "일"), (YR, 2629800, "개월")]:
        if s < lim: return f"{s/div:.0f}{u}"
    y = s / YR
    if y < 1e4: return f"{y:,.0f}년"
    return "10" + str(int(math.log10(y))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")) + "년"
def tick_axis(fig, axis="y"):
    kw = dict(tickvals=[math.log10(v) for v, _ in TICKS], ticktext=[t for _, t in TICKS])
    fig.update_yaxes(**kw) if axis == "y" else fig.update_xaxes(tickangle=0, **kw)
def style(fig, h=310):
    fig.update_layout(height=h, margin=dict(l=16, r=18, t=26, b=14), paper_bgcolor="#fff", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color="#1d1d1f"), legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)), hoverlabel=dict(font_family=FONT, bgcolor="white"))
    fig.update_xaxes(gridcolor="#ececf0", zeroline=False, showline=False); fig.update_yaxes(gridcolor="#ececf0", zeroline=False, showline=False)
    return fig
def show(fig):
    cfg = {"displayModeBar": False}
    try: st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError: st.plotly_chart(fig, use_container_width=True, config=cfg)
_bd = [0]
def head(title, take):
    b = ""
    if title.startswith("질문"): _bd[0] = 0
    else: _bd[0] += 1; b = f'<span class="bd">분석 {_bd[0]}</span>'
    md(f'<div class="sec"><div class="t">{b}{title}</div><div class="k">{take}</div></div>')
def source(key):
    n, t = SRC[key]
    with st.expander("데이터 출처 · 집계 기준 · 한계"): st.markdown(f"**{n}**\n\n{t}")
def kpis(items):
    cards = "".join(f'<div class="c" style="animation-delay:{i*90}ms"><div class="l">{l}</div><div class="v" style="font-size:{26 if isinstance(v,(int,float)) else 17}px"><span class="n" data-v="{v if isinstance(v,(int,float)) else ""}" data-d="{d}">{v if not isinstance(v,(int,float)) else f"{v:,.{d}f}"}</span><span class="u">{u}</span></div></div>' for i, (l, v, d, u) in enumerate(items))
    h = f"""<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" rel="stylesheet"><style>
    body{{margin:0;font-family:{FONT}}}.g{{display:grid;grid-template-columns:repeat({len(items)},1fr);gap:14px}}
    .c{{background:#fff;border-radius:18px;padding:14px 18px;border-left:5px solid #4cc38a;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:u .6s cubic-bezier(.2,.8,.2,1) both}}
    .l{{color:#6e6e73;font-size:13px;font-weight:600}}.v{{margin-top:4px;color:#1d1d1f;font-size:26px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.02em}}.u{{font-size:15px;color:#6e6e73;margin-left:3px}}
    @keyframes u{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}@media(max-width:700px){{.g{{grid-template-columns:1fr 1fr}}}}</style>
    <div class="g">{cards}</div><script>document.querySelectorAll('.n').forEach(e=>{{const t=parseFloat(e.dataset.v);if(isNaN(t))return;const d=+e.dataset.d,s=performance.now();
    const f=n=>{{const p=Math.min((n-s)/1100,1),x=1-Math.pow(1-p,3);e.textContent=(t*x).toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}});if(p<1)requestAnimationFrame(f)}};requestAnimationFrame(f)}})</script>"""
    try: components.html(h, height=104)
    except Exception: md("".join(f"<b>{l}</b> {v}{u} &nbsp; " for l, v, d, u in items))

@st.cache_data
def load():
    m = pd.read_csv(D / "metrics_unique.csv"); m["등급"] = pd.Categorical(m.class_strength.map(KO), ORDER, ordered=True)
    m["lc"] = np.log10(m.crack_time_sec.clip(lower=1e-12))
    p = pd.read_csv(D / "common_passwords.csv"); p["lon"] = np.log10(p.online_sec); p["lof"] = np.log10(p.offline_crack_sec)
    c = pd.read_csv(D / "cyber_synth.csv")
    return m, p, c
M, P, C = load()
TOP = set(P.password.str.lower())

md("""<style>@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@property --p{syntax:'<integer>';inherits:true;initial-value:0}
html,body,[class*="css"],.stApp{font-family:Pretendard,-apple-system,'Apple SD Gothic Neo',sans-serif!important}
.stApp{background:#e9edf2}#MainMenu,footer{visibility:hidden}[data-testid="stHeader"]{background:transparent}.block-container{max-width:1600px;padding:1rem 1.6rem 0;position:relative;z-index:1;animation:up .6s both}
.stApp::before,.stApp::after{content:"";position:fixed;width:560px;height:560px;border-radius:50%;z-index:0;pointer-events:none;filter:blur(30px)}
.stApp::before{top:-160px;right:-100px;background:radial-gradient(circle,#a9c8ff99,transparent 70%);animation:fl 13s ease-in-out infinite alternate}
.stApp::after{bottom:-200px;left:200px;background:radial-gradient(circle,#9fe8c299,transparent 70%);animation:fl 17s ease-in-out infinite alternate-reverse}
@keyframes fl{to{transform:translate(-80px,60px) scale(1.2)}}
[data-testid="stSidebar"]{background:#fff;border-radius:0 28px 28px 0;box-shadow:0 0 30px rgba(0,0,0,.06)}[data-testid="stSidebar"] label[data-baseweb="radio"]>div:first-child{display:none}
[data-testid="stSidebar"] label[data-baseweb="radio"]{padding:12px 14px;border-radius:14px;width:100%;margin:2px 0;transition:all .25s;font-weight:600;color:#4b5563}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{background:#f1f4f8;transform:translateX(4px)}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){background:#e5f7ee;color:#1e9e5a;box-shadow:inset 3px 0 0 #4cc38a}
.logo{font-size:26px;font-weight:900;letter-spacing:-.03em;background:linear-gradient(90deg,#4cc38a,#5b8def,#af52de);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 5s linear infinite alternate;margin:4px 0 18px}
.foot{color:#9ca3af;font-size:13px;margin-top:40px}@keyframes sh{to{background-position:100% 0}}
.nt{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:6px}.nt div{background:#fff;border-radius:16px;padding:14px 20px;display:flex;gap:14px;align-items:center;box-shadow:0 2px 12px rgba(0,0,0,.04);animation:up .6s both}
.nt b{font-size:14px}.nt span{color:#6b7280;font-size:14px;flex:1}.nt i{color:#b0b6c0;font-size:13px;font-style:normal}
.sec{margin:6px 0 4px}.sec .t{font-size:16px;font-weight:800;letter-spacing:-.02em}.sec .k{font-size:12.5px;color:#6b7280;margin-top:2px;line-height:1.4}
div[data-testid="stPlotlyChart"],.stPlotlyChart{background:#fff;border-radius:22px;padding:0;overflow:hidden;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:wipe 1.2s cubic-bezier(.2,.8,.2,1) both;transition:transform .3s,box-shadow .3s}
div[data-testid="stPlotlyChart"]:hover{transform:translateY(-4px);box-shadow:0 14px 34px rgba(60,90,140,.16)}
@keyframes up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}@keyframes wipe{from{opacity:0;clip-path:inset(0 100% 0 0 round 22px)}to{opacity:1;clip-path:inset(0 0 0 0 round 22px)}}
.rgs{display:flex;gap:12px}.rg{flex:1;min-width:0;height:104px;box-sizing:border-box;background:#fff;border-radius:20px;padding:10px 16px;display:flex;align-items:center;gap:12px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .7s both;transition:transform .3s}.rg:hover{transform:translateY(-4px)}
.rw{position:relative;width:80px;height:80px;flex:none;counter-reset:n var(--p);animation:fill 1.8s cubic-bezier(.2,.8,.2,1) .3s forwards}
.rw::after{content:counter(n) '%';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:17px;font-weight:800;color:var(--c)}
.ring{width:100%;height:100%;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*3.6deg),#eef1f5 0);-webkit-mask:radial-gradient(circle,transparent 31px,#000 32px);mask:radial-gradient(circle,transparent 31px,#000 32px)}
@keyframes fill{to{--p:var(--t)}}.rd b{font-size:14px}.rd span{color:#6b7280;font-size:12px;line-height:1.35}
.box{background:#fff;border-radius:22px;padding:22px 26px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .6s both;line-height:1.7}.box b{font-size:19px}
.pill{display:inline-block;background:#fff4e0;color:#b25e00;font-weight:700;font-size:13px;border-radius:99px;padding:5px 12px;margin:6px 0;animation:pu 2s infinite}@keyframes pu{50%{box-shadow:0 0 0 7px #ff9f0a22}}
.chip{display:inline-block;border-radius:99px;padding:5px 12px;margin:3px 4px 3px 0;font-size:14px;font-weight:600}.ok{background:#e3f8e9;color:#1b7a36}.no{background:#ffe9e7;color:#c4261b}
.spl{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.9s ease forwards}
.spl div{color:#fff;font-size:clamp(30px,6vw,64px);font-weight:800;letter-spacing:-.03em;animation:si 1.6s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.92);filter:blur(10px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,72%{opacity:1}100%{opacity:0;visibility:hidden}}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b0f1a 0%,#111a33 100%)!important;border-right:1px solid #25305a;box-shadow:0 0 40px rgba(0,0,0,.25)}
[data-testid="stSidebar"] [role="radiogroup"] label>div:not([data-testid="stMarkdownContainer"]){display:none!important}
[data-testid="stSidebar"] [role="radiogroup"] label{background:transparent;border:1px solid transparent;color:#c9d1e6!important}
[data-testid="stSidebar"] [role="radiogroup"] label p{color:#c9d1e6!important;font-size:15px}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#172044!important;border-color:#25305a}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(91,141,239,.28),rgba(91,141,239,.06))!important;border-color:#2f4286;box-shadow:inset 3px 0 0 #5b8def!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#8fb4ff!important;font-weight:700}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,[data-testid="stSidebar"] label p{color:#aab4d0}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#131c33!important;border:1px solid #25305a!important;color:#e6e9f2}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#e6e9f2}
[data-testid="stSidebar"] span[data-baseweb="tag"]{background:#5b8def!important;border-radius:8px}[data-testid="stSidebar"] span[data-baseweb="tag"] *{color:#fff!important}
[data-testid="stSidebar"] [role="slider"]{background:#5b8def!important}[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],[data-testid="stSidebar"] [data-testid="stTickBarMin"],[data-testid="stSidebar"] [data-testid="stTickBarMax"]{color:#8fb4ff!important}
[data-testid="stSidebar"] button[kind="header"] svg,[data-testid="stSidebarCollapseButton"] svg{color:#8d97b5!important;fill:#8d97b5!important}[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
.st-key-hdr{background:linear-gradient(100deg,#171f3d,#0f1730);border-radius:20px;padding:10px 20px;border:1px solid #25305a;box-shadow:0 6px 24px rgba(15,23,48,.25);animation:up .6s both}
.st-key-hdr *{color:#e6e9f2}.ht{font-size:19px;font-weight:800;letter-spacing:-.02em}.ht small{display:block;font-size:11px;color:#8d97b5!important;font-weight:500;margin-top:2px}
.st-key-hdr [role="radiogroup"]{flex-direction:row;gap:6px;flex-wrap:nowrap}.st-key-hdr [role="radiogroup"] label{padding:8px 14px;border-radius:12px;background:#1a2547;border:1px solid #2a3868;margin:0;transition:all .25s}
.st-key-hdr [role="radiogroup"] label>div:not([data-testid="stMarkdownContainer"]){display:none!important}.st-key-hdr label p{font-size:14px;font-weight:700;white-space:nowrap}
.st-key-hdr [role="radiogroup"] label:hover{transform:translateY(-2px);border-color:#5b8def}.st-key-hdr [role="radiogroup"] label:has(input:checked){background:#5b8def;border-color:#8fb4ff;box-shadow:0 0 18px rgba(91,141,239,.55)}
.st-key-hdr button{background:#1a2547;border:1px solid #2a3868;border-radius:12px;color:#e6e9f2}
.nc{background:#fff;border-radius:14px;padding:8px 14px;display:flex;gap:10px;align-items:center;line-height:1.35}.nc b{color:#1d1d1f!important;font-size:15px}.nc span{color:#6b7280!important;font-size:11.5px}
.bd{background:#5b8def;color:#fff;font-size:11px;font-weight:700;border-radius:6px;padding:2px 8px;margin-right:8px;vertical-align:2px}
.ins{background:#fff;border-left:5px solid #5b8def;border-radius:12px;padding:10px 16px;font-size:14px;box-shadow:0 2px 12px rgba(0,0,0,.05);animation:up .6s both;margin-top:4px}
span[data-baseweb="tag"]{background:#5b8def!important;border-radius:8px}[role="slider"]{background:#5b8def!important}
.gls{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:2px 0}.gls div{background:#fff;border-radius:14px;padding:9px 14px;font-size:12.5px;color:#4b5563;line-height:1.45;box-shadow:0 2px 12px rgba(0,0,0,.04);animation:up .6s both}.gls b{display:block;font-size:13.5px;color:#1d1d1f}[data-testid="stVerticalBlock"]{gap:.55rem}[data-testid="stExpander"]{border-radius:14px;background:#fff}@media(max-height:900px){.block-container{zoom:.92}}@media(max-height:820px){.block-container{zoom:.84}}@media(max-height:740px){.block-container{zoom:.76}}@media(max-height:660px){.block-container{zoom:.68}}@media(prefers-reduced-motion:reduce){*{animation-duration:.01s!important}}</style>""")
if not st.session_state.get("splash"):
    st.session_state["splash"] = True
    md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>')
NAMES = ["👋 시작하기", "📊 유출 현황", "⚠️ 흔한 비밀번호", "🧪 안전 조건", "🔍 내 진단"]
if "nav" not in st.session_state: st.session_state["nav"] = NAMES[0]
@st.fragment(run_every=12)
def tour():
    if st.session_state.get("tour_on") and time.time() - st.session_state.get("tour_t", 0) > 11:
        st.session_state["nav"] = NAMES[(NAMES.index(st.session_state["nav"]) + 1) % len(NAMES)]; st.session_state["tour_t"] = time.time(); st.rerun()
with st.container(key="hdr"):
    hc = st.columns([2.7, 5.6, 1.1, 0.95, 2.3], vertical_alignment="center")
    hc[0].markdown('<div class="ht">🔐 비밀번호 보안 분석 대시보드<small>PASSWORD SECURITY · 유출·흔한·합성 데이터 + 내 비밀번호 진단</small></div>', unsafe_allow_html=True)
    page = NAMES.index(hc[1].radio("메뉴", NAMES, horizontal=True, label_visibility="collapsed", key="nav"))
    hc[2].toggle("자동 순회", key="tour_on", on_change=lambda: st.session_state.update(tour_t=time.time()))
    hc[3].button("▶ 인트로", on_click=lambda: st.session_state.update(splash=False))
    hc[4].markdown('<div class="nc"><div style="font-size:24px">🎓</div><div><b>김시원</b><br><span>한국외대 Social Science &amp; AI · 산업데이터시각화</span></div></div>', unsafe_allow_html=True)
tour()
def ring(items):
    md('<div class="rgs">' + "".join(f'<div class="rg"><div class="rw" style="--t:{p:.0f};--c:{c}"><div class="ring"></div></div><div class="rd"><b>{l}</b><br><span>{d}</span></div></div>' for l, p, c, d in items) + "</div>")

if page == 0:
    head("질문 0. 비밀번호가 '뚫린다'는 건 무슨 뜻일까?", "처음 보시는 분을 위한 요약판입니다. 아래 기준 시간을 바꾸면 세 개의 와플 차트와 숫자가 함께 바뀝니다.")
    fl = st.columns([2, 2, 3], vertical_alignment="center")
    TS = {"1분": 60, "1시간": 3600, "1일": 86400, "1년": YR, "100년": YR * 100}
    tk = fl[0].select_slider("이 시간 안에 뚫리면 '위험'으로 셉니다", options=list(TS), value="1년")
    T = TS[tk]; fl[2].caption("↻ 기준 시간을 바꾸면 와플 차트와 숫자가 바뀝니다")
    md('<div class="gls"><div><b>🔓 크랙(뚫림)</b>프로그램이 가능한 비밀번호를 하나씩 대입해 맞는 것을 찾는 일입니다. 오래 걸릴수록 안전합니다.</div><div><b>🌐 온라인 vs 💾 오프라인</b>온라인은 로그인 창에 직접 시도(느림), 오프라인은 유출된 파일을 컴퓨터로 대량 대입(훨씬 빠름)하는 공격입니다.</div><div><b>🎲 엔트로피</b>비밀번호를 얼마나 예측하기 어려운지 나타내는 점수(bit)입니다. 클수록 안전합니다.</div></div>')
    pcts = [(M.crack_time_sec <= T).mean() * 100, (P.online_sec <= T).mean() * 100, (P.offline_crack_sec <= T).mean() * 100]
    kpis([("분석한 비밀번호", len(M), 0, "개"), ("흔한 비밀번호 목록", len(P), 0, "개"), ("합성 실험 표본", len(C), 0, "개"), (f"유출 표본 · {tk} 내 뚫림", pcts[0], 0, "%"), (f"흔한 PW 온라인 · {tk} 내", pcts[1], 0, "%"), (f"흔한 PW 오프라인 · {tk} 내", pcts[2], 0, "%")])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("100개 중 몇 개가 뚫릴까?(와플 차트)", f"네모 한 칸이 비밀번호 100개 중 1개이고, 빨강은 {tk} 안에 뚫리는 것입니다. 유출 표본은 등급별 균등 추출이라 실제 비율은 아닙니다.")
        ttl = ["유출 표본", "흔한 PW·온라인", "흔한 PW·오프라인"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t} {p:.0f}%" for t, p in zip(ttl, pcts)])
        for i, p in enumerate(pcts):
            z = np.zeros(100); z[:int(round(p))] = 1
            fig.add_trace(go.Heatmap(z=z.reshape(10, 10), colorscale=[[0, "#cfeedd"], [1, "#ff6b6b"]], showscale=False, xgap=3, ygap=3, hoverinfo="skip"), 1, i + 1)
            fig.update_xaxes(visible=False, row=1, col=i + 1); fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i + 1)
        fig.update_annotations(font_size=11); show(style(fig, 310))
    with x2:
        head("등급별 '보통 이 정도 걸려요'(범위 차트)", "막대는 가운데 50%의 비밀번호가 걸리는 시간 범위, 마름모는 중앙값입니다. 눈금은 10배씩 커지는 로그 눈금입니다.")
        gq = M.groupby("등급", observed=True).crack_time_sec.quantile([.25, .5, .75]).unstack(); fig = go.Figure()
        for o in ORDER:
            a_, m_, b_ = (math.log10(max(gq.loc[o, c_], 1e-6)) for c_ in (.25, .5, .75))
            fig.add_trace(go.Bar(y=[o], x=[b_ - a_], base=[a_], orientation="h", marker_color=CCOL[o], opacity=.75, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(y=[o], x=[m_], mode="markers+text", text=[human(10 ** m_)], textposition="top center", marker=dict(size=11, color="#1d1d1f", symbol="diamond"), showlegend=False, hoverinfo="skip"))
        tv = [1, 3600, 86400, YR, YR * 100, YR * 1e6, YR * 1e12, YR * 1e24]
        fig.update_xaxes(tickvals=[math.log10(v) for v in tv], ticktext=[human(v) for v in tv], tickangle=0); show(style(fig, 310))
    with x3:
        head("길이 → 강도 등급 흐름(산키 차트)", "선이 굵을수록 비밀번호가 많습니다. 길이 구간별로 어느 강도 등급이 되는지 따라가 보세요.")
        bins = ["≤7자", "8~11자", "12~15자", "16~20자", "21자+"]
        sk = M.assign(구간=pd.cut(M.length, [0, 7, 11, 15, 20, 70], labels=bins).astype(str), 등급=M.등급.astype(str)).groupby(["구간", "등급"]).size().reset_index(name="n")
        fig = go.Figure(go.Sankey(node=dict(label=bins + ORDER, color=["#9db7ee"] * 5 + [CCOL[o] for o in ORDER], pad=14, thickness=14), link=dict(source=[bins.index(a) for a in sk.구간], target=[5 + ORDER.index(b) for b in sk.등급], value=sk.n, color="rgba(91,141,239,.22)")))
        show(style(fig, 310))
    md(f'<div class="ins"><b>한 줄 요약</b> 같은 기준({tk})으로 보면 유출 표본의 {pcts[0]:.0f}%, 흔한 비밀번호는 온라인 {pcts[1]:.0f}%·오프라인 {pcts[2]:.0f}%가 뚫립니다. 어디서 공격받느냐에 따라 위험이 크게 달라집니다.</div>')
    with st.expander("데이터 출처 · 지도(지역) 시각화를 넣지 않은 이유"):
        st.markdown("세 데이터 모두 국가·지역 정보가 없어 지도 시각화는 넣지 않았습니다. 데이터에 없는 위치를 임의로 만들어 붙이는 것은 정직한 시각화가 아니라고 판단했습니다. 데이터별 출처와 집계 기준은 각 페이지 하단에 있습니다.")
if page == 1:
    head("질문 1. 길이가 늘면 뚫는 시간은 얼마나 달라질까?", "고유 비밀번호 25,000개의 길이·엔트로피·크랙 시간을 조건별로 살펴봅니다.")
    fl = st.columns([2, 2, 3], vertical_alignment="center")
    fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")
    gs = fl[0].multiselect("강도 등급", ORDER, default=ORDER)
    lr = fl[1].slider("길이 범위(자)", 4, 63, (4, 40))
    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty: st.info("조건에 맞는 데이터가 없습니다. 등급이나 길이 범위를 넓혀 보세요."); st.stop()
    kpis([("고유 비밀번호", len(f), 0, "개"), ("길이 중앙값", float(f.length.median()), 0, "자"), ("엔트로피 중앙값", float(f.entropy.median()), 1, "bit"), ("크랙 시간 중앙값", human(f.crack_time_sec.median()), 0, "")] + [(_l, _p, 0, "%") for _l, _p, _c, _d in [("1년 안에 뚫림", (f.crack_time_sec <= YR).mean() * 100, "#ff6b6b", "선택한 비밀번호 중 1년 이내 크랙 비율"), ("1시간 안에 뚫림", (f.crack_time_sec <= 3600).mean() * 100, "#ff9f0a", "선택한 비밀번호 중 1시간 이내 크랙 비율")]])
    g = f.groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=20").reset_index()
    slope = np.polyfit(f.length, f.lc, 1)[0] if f.length.nunique() > 1 else float("nan")
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("길이별 크랙 시간(중앙값)", f"길이가 1자 늘 때마다 크랙 시간은 평균 약 {10**slope:.1f}배 늘어납니다(선택 구간 기준). 세로축은 로그 눈금이며 표본 20개 미만인 길이는 제외했습니다." if np.isfinite(slope) else "길이 구간을 넓히면 추세가 보입니다.")
        fig = go.Figure(go.Scatter(x=g.length, y=np.log10(g.med), mode="lines+markers", line=dict(color=BLUE, width=3, shape="spline"), marker=dict(size=8), customdata=[[human(v), n] for v, n in zip(g.med, g.n)],
            hovertemplate="길이 %{x}자<br>중앙값 %{customdata[0]}<br>표본 %{customdata[1]}개<extra></extra>"))
        fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=GRAY, annotation_text="1년", annotation_position="bottom right")
        fig.update_xaxes(title="비밀번호 길이(자)"); tick_axis(fig); show(style(fig))
        with st.popover("📋 표 보기"): st.dataframe(g.assign(중앙값=g.med.map(human))[["length", "n", "중앙값"]].rename(columns={"length": "길이", "n": "표본"}), hide_index=True)
    with x2:
        head("등급·길이 구성(트리맵)", "선택한 비밀번호가 등급과 길이 구간으로 어떻게 나뉘는지 면적으로 봅니다. 등급별 5,000개 균등 표본이라 등급 면적이 비슷한 것이 정상입니다.")
        tm = f.assign(구간=pd.cut(f.length, [0, 7, 11, 15, 20, 70], labels=["≤7자", "8~11자", "12~15자", "16~20자", "21자+"]).astype(str), 등급=f.등급.astype(str)).groupby(["등급", "구간"]).size().reset_index(name="n")
        fig = px.treemap(tm, path=["등급", "구간"], values="n", color="등급", color_discrete_map=CCOL)
        fig.update_traces(textinfo="label+value", marker_line_width=2); show(style(fig, 310))
    with x3:
        head("엔트로피와 크랙 시간", f"엔트로피–크랙 시간 상관 r={np.corrcoef(f.entropy, f.lc)[0,1]:.2f}. 엔트로피가 길이에서 거의 그대로 계산돼 점들이 한 곡선을 따릅니다." if len(f) > 2 else "")
        s = f.sample(min(2500, len(f)), random_state=7)
        fig = go.Figure([go.Scatter(x=d.entropy, y=d.lc, mode="markers", name=k, marker=dict(size=5, color=CCOL[k], opacity=.55)) for k, d in s.groupby("등급", observed=True)])
        fig.update_xaxes(title="엔트로피(bit)"); tick_axis(fig); show(style(fig, 310))
    md(f'<div class="ins"><b>핵심 결론</b> 선택한 {len(f):,}개 중 {(f.crack_time_sec <= YR).mean():.0%}가 1년 안에 뚫립니다. 길이 중앙값 {int(f.length.median())}자 비밀번호의 크랙 시간 중앙값은 {human(f.crack_time_sec.median())}입니다.</div>')
    source("t1")

if page == 2:
    head("질문 2. 가장 흔한 비밀번호는 얼마나 빨리 뚫릴까?", "흔한 비밀번호 500개를 순위·카테고리·공격 방식별로 비교합니다.")
    fl = st.columns([2, 2, 3], vertical_alignment="center")
    fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")
    n = fl[0].slider("상위 N개", 5, 100, 20, 5)
    cats = fl[1].multiselect("카테고리", sorted(P.category.unique()), default=sorted(P.category.unique()))
    q = P[(P["rank"] <= n) & P.category.isin(cats)]
    if q.empty: st.info("조건에 맞는 비밀번호가 없습니다. 카테고리를 선택해 주세요."); st.stop()
    topc = q.category.value_counts()
    kpis([("표시 중 비밀번호", len(q), 0, "개"), ("가장 많은 카테고리", f"{topc.index[0]}", 0, f" {topc.iloc[0]/len(q):.0%}"), ("온라인 크랙 중앙값", human(q.online_sec.median()), 0, ""), ("오프라인 크랙 중앙값", human(q.offline_crack_sec.median()), 0, "")] + [(_l, _p, 0, "%") for _l, _p, _c, _d in [("온라인 1일 이내", (q.online_sec <= 86400).mean() * 100, "#5b8def", "온라인 공격으로 하루 안에 뚫리는 비율"), ("오프라인 1초 이내", (q.offline_crack_sec <= 1).mean() * 100, "#ff6b6b", "오프라인 공격으로 1초 안에 뚫리는 비율")]])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head(f"인기 순위 TOP {min(n, len(q))}", "점이 왼쪽에 있을수록 빨리 뚫립니다. 로그 눈금이라 한 칸이 10배 차이입니다.")
        qq = q.head(20)
        fig = go.Figure()
        for k, d in qq.groupby("category"): fig.add_trace(go.Scatter(x=d.lon, y=[f"{int(r)}위 {w}" for r, w in zip(d["rank"], d.password)], mode="markers", name=k, marker=dict(size=11),
            customdata=[human(v) for v in d.online_sec], hovertemplate="%{y}<br>온라인 %{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[f"{int(r)}위 {w}" for r, w in zip(qq["rank"], qq.password)]); tick_axis(fig, "x")
        fig.update_xaxes(title="온라인 공격 크랙 시간(로그)"); show(style(fig, 310).update_layout(showlegend=False))
        with st.popover("📋 표 보기"): st.dataframe(qq.assign(온라인=qq.online_sec.map(human), 오프라인=qq.offline_crack_sec.map(human))[["rank", "password", "category", "온라인", "오프라인"]].rename(columns={"rank": "순위", "password": "비밀번호", "category": "카테고리"}), hide_index=True)
        if len(q) > 20: st.caption("순위 차트는 상위 20개만 표시합니다.")
    with x2:
        head("카테고리·흔한 비밀번호 한눈에", f"가장 많은 유형은 '{topc.index[0]}'({topc.iloc[0]}개)입니다. 버블이 클수록 순위가 높은 비밀번호입니다.")
        tA, tB = st.tabs(["선버스트", "버블 클라우드"])
        with tA:
            sb = q.assign(구간=pd.cut(q["rank"], [0, 10, 25, 50, 100, 500], labels=["1~10위", "11~25위", "26~50위", "51~100위", "101위+"]).astype(str)).groupby(["category", "구간"]).size().reset_index(name="n")
            fig = px.sunburst(sb, path=["category", "구간"], values="n", color_discrete_sequence=px.colors.qualitative.Pastel); show(style(fig, 262))
        with tB:
            bq = q.head(30).reset_index(drop=True); i_ = np.arange(len(bq)); bq["x"] = 21 * np.sqrt(i_ + .5) * np.cos(i_ * 2.39996); bq["y"] = 21 * np.sqrt(i_ + .5) * np.sin(i_ * 2.39996)
            pal = dict(zip(sorted(bq.category.unique()), px.colors.qualitative.Set3 * 2)); fig = go.Figure()
            for cat, d_ in bq.groupby("category"):
                fig.add_trace(go.Scatter(x=d_.x, y=d_.y, mode="markers+text", text=d_.password, textfont=dict(size=9), name=cat, marker=dict(size=16 + 26 / np.sqrt(d_["rank"]), color=pal[cat], opacity=.9, line=dict(color="#fff", width=1)), customdata=d_["rank"], hovertemplate="%{text}<br>%{customdata}위<extra></extra>"))
            fig.update_xaxes(visible=False, range=[-175, 175]); fig.update_yaxes(visible=False, range=[-135, 135]); fig.update_layout(showlegend=False); show(style(fig, 262))
    with x3:
        gap = (q.lon - q.lof).median()
        head("온라인 vs 오프라인(박스플롯)", f"오프라인 공격은 온라인보다 중앙값 기준 약 10^{gap:.0f}배 빠릅니다. 상자는 카테고리 안의 분포입니다.")
        fig = go.Figure([go.Box(x=q.lon, y=q.category, name="온라인", orientation="h", marker_color=BLUE, boxpoints=False), go.Box(x=q.lof, y=q.category, name="오프라인", orientation="h", marker_color=RED, boxpoints=False)])
        fig.update_layout(boxmode="group"); tick_axis(fig, "x"); show(style(fig, 310))
    md(f'<div class="ins"><b>핵심 결론</b> 표시 중 {len(q)}개 중 {(q.online_sec <= 86400).mean():.0%}는 온라인 공격에도 하루 안에 뚫리고, 오프라인 공격은 중앙값 {human(q.offline_crack_sec.median())}이면 충분합니다.</div>')
    source("t2")

if page == 3:
    head("질문 3. 어떤 조건이 비밀번호를 강하게 만들까?", "길이와 문자 구성(대문자·숫자·특수문자)이 강도 점수와 침해 위험도에 주는 영향을 봅니다.")
    md('<span class="pill">합성 데이터 기반 탐색</span>')
    fl = st.columns([2, 2, 3], vertical_alignment="center")
    fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")
    rk = fl[0].multiselect("침해 위험도", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    lr3 = fl[1].slider("길이 범위(자)", 5, 20, (5, 20), key="l3")
    h = C[C.Breach_Risk.isin(rk) & C.Password_Length.between(*lr3)]
    if h.empty: st.info("조건에 맞는 데이터가 없습니다."); st.stop()
    kpis([("표본", len(h), 0, "개"), ("평균 강도 점수", float(h.Strength_Score.mean()), 1, "점"), ("High 위험 비율", float((h.Breach_Risk == "High").mean() * 100), 1, "%"), ("길이–점수 상관", float(h.Password_Length.corr(h.Strength_Score)) if len(h) > 2 else 0.0, 2, "")] + [(_l, _p, 0, "%") for _l, _p, _c, _d in [("High 위험", (h.Breach_Risk == "High").mean() * 100, "#ff6b6b", "침해 위험이 높은 비밀번호 비율"), ("Low 위험", (h.Breach_Risk == "Low").mean() * 100, "#4cc38a", "침해 위험이 낮은 비밀번호 비율")]])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("길이 × 구성 종류 → 평균 점수(히트맵)", "가로는 길이, 세로는 포함한 문자 종류(대문자·숫자·특수문자) 수입니다. 초록일수록 강합니다.")
        k = h.assign(종류=(h.Uppercase_Count > 0).astype(int) + (h.Digit_Count > 0) + (h.Special_Char_Count > 0))
        pv = k.pivot_table(index="종류", columns="Password_Length", values="Strength_Score", aggfunc="mean").sort_index()
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=[f"{i}종" for i in pv.index], colorscale=[[0, "#ffe9e7"], [.5, "#fff2dc"], [1, "#4cc38a"]], zmin=0, zmax=100, colorbar=dict(title="점수", thickness=10), hovertemplate="길이 %{x}자 · %{y}<br>평균 %{z:.0f}점<extra></extra>"))
        fig.update_xaxes(title="길이(자)"); show(style(fig, 310))
        with st.popover("📋 표 보기"): st.dataframe(pv.round(1).rename(columns=str).reset_index(), hide_index=True)
    with x2:
        head("구성 요소별 평균 점수(레이더)", "포함 여부별 평균 점수를 겹쳤습니다. 길이 효과가 섞여 있으니 길이 범위를 좁혀 비교하세요.")
        nm = ["대문자", "숫자", "특수문자"]; cl = ["Uppercase_Count", "Digit_Count", "Special_Char_Count"]
        inc = [h[h[c] > 0].Strength_Score.mean() for c in cl]; exc = [h[h[c] == 0].Strength_Score.mean() for c in cl]
        fig = go.Figure([go.Scatterpolar(r=inc + inc[:1], theta=nm + nm[:1], fill="toself", name="포함", line_color=BLUE), go.Scatterpolar(r=exc + exc[:1], theta=nm + nm[:1], fill="toself", name="미포함", line_color=GRAY)])
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#ececf0"), bgcolor="rgba(0,0,0,0)")); show(style(fig, 310))
    with x3:
        head("점수 분포와 침해 위험(히스토그램)", "위험도는 점수 구간으로 정해진 값이라 색이 깔끔하게 갈립니다.")
        fig = go.Figure([go.Histogram(x=h[h.Breach_Risk == r].Strength_Score, name=r, marker_color=RCOL[r], xbins=dict(start=0, end=100, size=5)) for r in ["High", "Medium", "Low"]])
        fig.update_layout(barmode="stack"); fig.update_xaxes(title="강도 점수"); fig.update_yaxes(title="개수"); show(style(fig, 310))
    md(f'<div class="ins"><b>핵심 결론</b> 평균 점수 {h.Strength_Score.mean():.0f}점, High 위험 {(h.Breach_Risk == "High").mean():.0%}. 합성 데이터라 길이가 점수를 거의 결정하는 구조임을 감안해 해석하세요.</div>')
    source("t3")

def pool_size(p): return 26 * bool(re.search("[a-z]", p)) + 26 * bool(re.search("[A-Z]", p)) + 10 * bool(re.search("[0-9]", p)) + 33 * bool(re.search(r"[ -/:-@\[-`{-~]", p)) + 100 * bool(re.search(r"[^\x00-\x7f]", p))
SEQ = ["abcdefghijklmnopqrstuvwxyz", "0123456789", "qwertyuiopasdfghjklzxcvbnm"]
@st.cache_data(ttl=3600, show_spinner=False)
def hibp_range(prefix):
    r = requests.get(f"https://api.pwnedpasswords.com/range/{prefix}", headers={"Add-Padding": "true", "User-Agent": "streamlit-password-dashboard"}, timeout=6); r.raise_for_status(); return r.text
def diagnose(pw, use_hibp):
    L = len(pw); low = pw.lower(); pool = pool_size(pw); ent = L * math.log2(pool) if pool else 0; notes = []
    seq = any(s[i:i+3] in low or s[i:i+3][::-1] in low for s in SEQ for i in range(len(s) - 2)); rep = bool(re.search(r"(.)\1\1", pw)); yr = bool(re.search(r"(19|20)\d\d", pw))
    ent -= 15 * seq + 10 * rep + 8 * yr
    cnt = None
    if use_hibp:
        try:
            sha = hashlib.sha1(pw.encode()).hexdigest().upper()
            cnt = next((int(c) for h, _, c in (x.partition(":") for x in hibp_range(sha[:5]).splitlines()) if h == sha[5:]), 0)
        except Exception: cnt = -1
    score = max(0, min(100, ent / 80 * 100)); common = low in TOP
    if common: score = min(score, 5)
    if cnt and cnt > 0: score = min(score, 10)
    st.session_state["diag"] = dict(L=L, score=score, ent=max(ent, 0), cnt=cnt, common=common, seq=seq, rep=rep, yr=yr, up=bool(re.search("[A-Z]", pw)), dg=bool(re.search("[0-9]", pw)), sp=bool(re.search(r"[^A-Za-z0-9]", pw)), allnum=pw.isdigit())

if page == 4:
    head("질문 4. 내 비밀번호는 안전할까?", "입력한 비밀번호는 저장하지 않으며, 유출 확인에는 해시 앞 5글자만 전송합니다(k-익명성).")
    ex = st.columns([1, 1, 1, 1.3])
    u = ex[3].checkbox("유출 여부 확인(Have I Been Pwned)", value=True)
    for col, (lab, v) in zip(ex[:3], [("예시 · 흔한 비밀번호", "password123"), ("예시 · 규칙형", "Summer2024!"), ("예시 · 긴 무작위", "vK9#mQ2$zL7@pW4xR8")]):
        if col.button(lab): diagnose(v, u)
    with st.form("f", clear_on_submit=True, border=False):
        fc = st.columns([6, 1])
        pw = fc[0].text_input("비밀번호 입력", type="password", placeholder="비밀번호를 입력하세요(저장되지 않습니다)", label_visibility="collapsed")
        if fc[1].form_submit_button("진단하기") and pw: diagnose(pw, u)
    d = st.session_state.get("diag")
    if d:
        s = d["score"]; grade, col = ("위험", RED) if s < 30 else ("주의", ORANGE) if s < 55 else ("양호", BLUE) if s < 80 else ("안전", GREEN)
        if d["cnt"] and d["cnt"] > 0: grade, col = "위험", RED
        c1, c2, c3 = st.columns([2, 3, 3])
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(s), number=dict(suffix="점", font=dict(size=44)), title=dict(text=f"위험도 등급: {grade}", font=dict(size=18)),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.35), bgcolor="#f2f2f5", borderwidth=0, steps=[dict(range=[0, 30], color="#ffe9e7"), dict(range=[30, 55], color="#fff2dc"), dict(range=[55, 80], color="#e5f1ff"), dict(range=[80, 100], color="#e3f8e9")])))
            show(style(fig, 230))
            vf = go.Figure([go.Violin(x=M[M.등급 == g_].length, name=g_, orientation="h", line_color=CCOL[g_], points=False, showlegend=False, spanmode="hard") for g_ in ORDER])
            vf.add_vline(x=d["L"], line_color="#1d1d1f", line_dash="dash", annotation_text="내 길이")
            vf.update_xaxes(title="탭 1 데이터의 등급별 길이 분포(자)"); show(style(vf, 210))
        with c2:
            chips = [("8자 이상", d["L"] >= 8), ("12자 이상", d["L"] >= 12), ("대문자", d["up"]), ("숫자", d["dg"]), ("특수문자", d["sp"]), ("흔한 목록 아님", not d["common"]), ("연속 패턴 없음", not d["seq"]), ("반복 문자 없음", not d["rep"]), ("연도 패턴 없음", not d["yr"])]
            ch = "".join(f'<span class="chip {"ok" if v else "no"}">{"✓" if v else "✕"} {k}</span>' for k, v in chips)
            br = {None: "유출 확인을 사용하지 않았습니다.", -1: "유출 확인 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요."}.get(d["cnt"], f"유출 기록 <b>{d['cnt']:,}회</b> 발견. 이미 공개된 비밀번호이니 즉시 교체해야 합니다." if (d["cnt"] or 0) > 0 else "유출 기록에서 발견되지 않았습니다(안전하다는 뜻은 아닙니다).")
            pct = (M.drop_duplicates("length").length.lt(0).sum(), (M.length < d["L"]).mean())[1]
            sub = C[C.Password_Length == min(max(d["L"], 5), 20)]
            md(f'<div class="box"><b>진단 결과</b><br>{ch}<br>추정 엔트로피 {d["ent"]:.0f}bit · 오프라인 무차별 대입(초당 100억 회 가정) 약 {human(2**d["ent"]/2/1e10)}<br>{br}<br>탭 1 데이터에서 내 길이({d["L"]}자)는 {pct:.0%}의 비밀번호보다 깁니다.' + (f" 탭 3 데이터의 같은 길이 표본은 High 위험이 {(sub.Breach_Risk=='High').mean():.0%}입니다." if len(sub) else "") + "</div>")
        tips = []
        if d["cnt"] and d["cnt"] > 0 or d["common"]: tips.append("이미 알려진 비밀번호입니다. 지금 사용 중이라면 바로 바꾸세요.")
        if d["L"] < 12: tips.append("12자 이상, 가능하면 16자 이상으로 늘리세요. 길이가 가장 큰 영향을 줍니다.")
        if d["seq"] or d["rep"] or d["yr"] or d["allnum"]: tips.append("연속 문자, 반복 문자, 연도, 숫자만으로 된 구성은 피하세요.")
        if not (d["up"] and d["dg"] and d["sp"]): tips.append("대문자·숫자·특수문자를 섞기보다 서로 관련 없는 단어 4개 이상을 이은 문장형 비밀번호가 기억하기 쉽고 강합니다.")
        tips += ["사이트마다 다른 비밀번호를 쓰고 비밀번호 관리자를 사용하세요.", "가능한 모든 계정에 2단계 인증을 켜세요."]
        with c3:
            md('<div class="box" style="margin-top:0"><b>개선 권고</b><br>' + "<br>".join(f"• {t}" for t in tips) + "</div>")
    with st.expander("데이터 출처 · 계산 기준 · 한계"):
        st.markdown("**유출 확인**: Have I Been Pwned의 Pwned Passwords range API(`api.pwnedpasswords.com/range`, 키 불필요)를 사용합니다. SHA-1 해시의 앞 5글자만 보내고 나머지는 브라우저가 아닌 앱 서버에서 대조하며, 비밀번호 원문은 저장하지 않습니다. 계정(이메일) 유출 조회용 v3 API는 유료 키가 필요해 사용하지 않았습니다.\n\n**점수**: 길이×log2(문자 종류 수)로 추정한 엔트로피에 패턴 감점을 적용해 80bit를 100점으로 환산했고, 유출·흔한 비밀번호는 상한을 둡니다. 교육용 근사치이며 실제 안전성을 보증하지 않습니다.")
