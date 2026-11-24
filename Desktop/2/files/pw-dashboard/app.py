import hashlib, math, re
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, requests, streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="비밀번호는 얼마나 버틸까?", page_icon="🔐", layout="wide", initial_sidebar_state="expanded")
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
def style(fig, h=220):
    fig.update_layout(height=h, margin=dict(l=16, r=18, t=26, b=14), paper_bgcolor="#fff", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color="#1d1d1f"), legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)), hoverlabel=dict(font_family=FONT, bgcolor="white"))
    fig.update_xaxes(gridcolor="#ececf0", zeroline=False, showline=False); fig.update_yaxes(gridcolor="#ececf0", zeroline=False, showline=False)
    return fig
def show(fig):
    cfg = {"displayModeBar": False}; height = int(fig.layout.height or 220)
    try: st.plotly_chart(fig, width="stretch", height=height, config=cfg)
    except TypeError: st.plotly_chart(fig, use_container_width=True, height=height, config=cfg)
def head(title, take): md(f'<div class="sec"><div class="t">{title}</div><div class="k">{take}</div></div>')
def source(key):
    n, t = SRC[key]
    with st.expander("데이터 출처 · 집계 기준 · 한계"): st.markdown(f"**{n}**\n\n{t}")
def kpis(items):
    cards = "".join(f'<div class="c" style="animation-delay:{i*90}ms"><div class="l">{l}</div><div class="v" style="font-size:{26 if isinstance(v,(int,float)) else 17}px"><span class="n" data-v="{v if isinstance(v,(int,float)) else ""}" data-d="{d}">{v if not isinstance(v,(int,float)) else f"{v:,.{d}f}"}</span><span class="u">{u}</span></div></div>' for i, (l, v, d, u) in enumerate(items))
    h = f"""<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" rel="stylesheet"><style>
    body{{margin:0;font-family:{FONT}}}.g{{display:grid;grid-template-columns:repeat({len(items)},1fr);gap:14px}}
    .c{{background:#fff;border-radius:18px;padding:14px 18px;border-left:5px solid #4cc38a;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:u .6s cubic-bezier(.2,.8,.2,1) both}}
    .l{{color:#6e6e73;font-size:13px;font-weight:600}}.v{{margin-top:4px;color:#1d1d1f;font-size:26px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.02em}}.u{{font-size:15px;color:#6e6e73;margin-left:3px}}
    @keyframes u{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
    @media(max-width:700px){{.g{{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.c{{padding:5px 8px;border-left-width:3px;border-radius:12px}}.l{{font-size:10px;line-height:1.15}}.v{{font-size:15px!important;line-height:1.2;margin-top:1px}}.u{{font-size:11px;margin-left:2px}}}}</style>
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
.rgs{display:flex;gap:12px;container-type:inline-size}.rg{flex:1;min-width:0;height:104px;box-sizing:border-box;background:#fff;border-radius:20px;padding:10px 16px;display:flex;align-items:center;gap:12px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .7s both;transition:transform .3s}.rg:hover{transform:translateY(-4px)}
.rw{position:relative;width:80px;height:80px;flex:none;counter-reset:n var(--p);animation:fill 1.8s cubic-bezier(.2,.8,.2,1) .3s forwards}
.rw::after{content:counter(n) '%';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:17px;font-weight:800;color:var(--c)}
.ring{width:100%;height:100%;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*3.6deg),#eef1f5 0);-webkit-mask:radial-gradient(circle,transparent 31px,#000 32px);mask:radial-gradient(circle,transparent 31px,#000 32px)}
@keyframes fill{to{--p:var(--t)}}.rd{min-width:0}.rd b{font-size:14px}.rd span{color:#6b7280;font-size:12px;line-height:1.35;word-break:keep-all}
@container(max-width:650px){.rgs{gap:8px}.rg{padding:8px;gap:8px;height:88px}.rw{width:58px;height:58px}.rw::after{font-size:14px}.ring{-webkit-mask:radial-gradient(circle,transparent 22px,#000 23px);mask:radial-gradient(circle,transparent 22px,#000 23px)}.rd b{display:block;font-size:11px;line-height:1.15}.rd span{display:none}}
.box{background:#fff;border-radius:22px;padding:22px 26px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .6s both;line-height:1.7}.box b{font-size:19px}
[data-testid="stForm"]{background:rgba(255,255,255,.94);border:1px solid rgba(255,255,255,.9);border-radius:20px;padding:22px 24px;box-shadow:0 8px 28px rgba(44,62,90,.08)}
[data-testid="stForm"] [data-testid="stTextInput"] input{min-height:48px;font-size:16px}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button{min-height:46px;font-weight:700}
.pill{display:inline-block;background:#fff4e0;color:#b25e00;font-weight:700;font-size:13px;border-radius:99px;padding:5px 12px;margin:6px 0;animation:pu 2s infinite}@keyframes pu{50%{box-shadow:0 0 0 7px #ff9f0a22}}
.chip{display:inline-block;border-radius:99px;padding:5px 12px;margin:3px 4px 3px 0;font-size:14px;font-weight:600}.ok{background:#e3f8e9;color:#1b7a36}.no{background:#ffe9e7;color:#c4261b}
.spl{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.9s ease forwards}
.spl div{color:#fff;font-size:clamp(30px,6vw,64px);font-weight:800;letter-spacing:-.03em;animation:si 1.6s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.92);filter:blur(10px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,72%{opacity:1}100%{opacity:0;visibility:hidden}}
[data-testid="stVerticalBlock"]{gap:.55rem}[data-testid="stExpander"]{border-radius:14px;background:#fff}@media(prefers-reduced-motion:reduce){*{animation-duration:.01s!important}}</style>""")
if not st.session_state.get("splash"):
    st.session_state["splash"] = True
    md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>')
NAMES = ["📊  유출 비밀번호 현황", "⚠️  흔한 비밀번호의 위험", "🧪  안전한 비밀번호의 조건", "🔍  내 비밀번호 진단"]
with st.sidebar:
    md('<div class="logo">PW ATLAS</div>')
    page = NAMES.index(st.radio("메뉴", NAMES, label_visibility="collapsed"))
def ring(items):
    md('<div class="rgs">' + "".join(f'<div class="rg"><div class="rw" style="--t:{p:.0f};--c:{c}"><div class="ring"></div></div><div class="rd"><b>{l}</b><br><span>{d}</span></div></div>' for l, p, c, d in items) + "</div>")

if page == 0:
    head("질문 1. 길이가 늘면 뚫는 시간은 얼마나 달라질까?", "고유 비밀번호 25,000개의 길이·엔트로피·크랙 시간을 조건별로 살펴봅니다.")
    gs = st.sidebar.multiselect("강도 등급", ORDER, default=ORDER)
    lr = st.sidebar.slider("길이 범위(자)", 4, 63, (4, 40))
    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty: st.info("조건에 맞는 데이터가 없습니다. 등급이나 길이 범위를 넓혀 보세요."); st.stop()
    r1, r2 = st.columns([5, 3])
    with r1:
        kpis([("고유 비밀번호", len(f), 0, "개"), ("길이 중앙값", float(f.length.median()), 0, "자"), ("엔트로피 중앙값", float(f.entropy.median()), 1, "bit"), ("크랙 시간 중앙값", human(f.crack_time_sec.median()), 0, "")])
    with r2:
        ring([("1년 안에 뚫림", (f.crack_time_sec <= YR).mean() * 100, "#ff6b6b", "선택한 비밀번호 중 1년 이내 크랙 비율"), ("1시간 안에 뚫림", (f.crack_time_sec <= 3600).mean() * 100, "#ff9f0a", "선택한 비밀번호 중 1시간 이내 크랙 비율")])
    g = f.groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=20").reset_index()
    slope = np.polyfit(f.length, f.lc, 1)[0] if f.length.nunique() > 1 else float("nan")
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("길이별 크랙 시간(중앙값)", f"길이가 1자 늘 때마다 크랙 시간은 평균 약 {10**slope:.1f}배 늘어납니다(선택 구간 기준). 세로축은 로그 눈금이며 표본 20개 미만인 길이는 제외했습니다." if np.isfinite(slope) else "길이 구간을 넓히면 추세가 보입니다.")
        fig = go.Figure(go.Scatter(x=g.length, y=np.log10(g.med), mode="lines+markers", line=dict(color=BLUE, width=3, shape="spline"), marker=dict(size=8), customdata=[[human(v), n] for v, n in zip(g.med, g.n)],
            hovertemplate="길이 %{x}자<br>중앙값 %{customdata[0]}<br>표본 %{customdata[1]}개<extra></extra>"))
        fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=GRAY, annotation_text="1년", annotation_position="bottom right")
        fig.update_xaxes(title="비밀번호 길이(자)"); tick_axis(fig); show(style(fig))
    with x2:
        head("강도 등급 분포", "등급별 5,000개씩 뽑은 균등 표본이라 막대 높이는 실제 비율이 아닙니다.")
        cnt = f.등급.value_counts().reindex(ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=ORDER, y=cnt.values, marker_color=[CCOL[o] for o in ORDER], text=[f"{int(v):,}" for v in cnt.values], textposition="outside", cliponaxis=False))
        fig.update_yaxes(title="비밀번호 수(개)"); show(style(fig))
    with x3:
        head("엔트로피와 크랙 시간", f"엔트로피–크랙 시간 상관 r={np.corrcoef(f.entropy, f.lc)[0,1]:.2f}. 엔트로피가 길이에서 거의 그대로 계산돼 점들이 한 곡선을 따릅니다." if len(f) > 2 else "")
        s = f.sample(min(2500, len(f)), random_state=7)
        fig = go.Figure([go.Scatter(x=d.entropy, y=d.lc, mode="markers", name=k, marker=dict(size=5, color=CCOL[k], opacity=.55)) for k, d in s.groupby("등급", observed=True)])
        fig.update_xaxes(title="엔트로피(bit)"); tick_axis(fig); show(style(fig))
    source("t1")

if page == 1:
    head("질문 2. 가장 흔한 비밀번호는 얼마나 빨리 뚫릴까?", "흔한 비밀번호 500개를 순위·카테고리·공격 방식별로 비교합니다.")
    n = st.sidebar.slider("상위 N개", 5, 100, 20, 5)
    cats = st.sidebar.multiselect("카테고리", sorted(P.category.unique()), default=sorted(P.category.unique()))
    q = P[(P["rank"] <= n) & P.category.isin(cats)]
    if q.empty: st.info("조건에 맞는 비밀번호가 없습니다. 카테고리를 선택해 주세요."); st.stop()
    topc = q.category.value_counts()
    r1, r2 = st.columns([5, 3])
    with r1:
        kpis([("표시 중 비밀번호", len(q), 0, "개"), ("가장 많은 카테고리", f"{topc.index[0]}", 0, f" {topc.iloc[0]/len(q):.0%}"), ("온라인 크랙 중앙값", human(q.online_sec.median()), 0, ""), ("오프라인 크랙 중앙값", human(q.offline_crack_sec.median()), 0, "")])
    with r2:
        ring([("온라인 1일 이내", (q.online_sec <= 86400).mean() * 100, "#5b8def", "온라인 공격으로 하루 안에 뚫리는 비율"), ("오프라인 1초 이내", (q.offline_crack_sec <= 1).mean() * 100, "#ff6b6b", "오프라인 공격으로 1초 안에 뚫리는 비율")])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head(f"인기 순위 TOP {min(n, len(q))}", "점이 왼쪽에 있을수록 빨리 뚫립니다. 로그 눈금이라 한 칸이 10배 차이입니다.")
        qq = q.head(20)
        fig = go.Figure()
        for k, d in qq.groupby("category"): fig.add_trace(go.Scatter(x=d.lon, y=[f"{int(r)}위 {w}" for r, w in zip(d["rank"], d.password)], mode="markers", name=k, marker=dict(size=11),
            customdata=[human(v) for v in d.online_sec], hovertemplate="%{y}<br>온라인 %{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[f"{int(r)}위 {w}" for r, w in zip(qq["rank"], qq.password)]); tick_axis(fig, "x")
        fig.update_xaxes(title="온라인 공격 크랙 시간(로그)"); show(style(fig).update_layout(showlegend=False))
        if len(q) > 20: st.caption("순위 차트는 상위 20개만 표시합니다.")
    with x2:
        head("카테고리별 분포", f"가장 많은 유형은 '{topc.index[0]}'({topc.iloc[0]}개)입니다.")
        fig = go.Figure(go.Bar(y=topc.index[::-1], x=topc.values[::-1], orientation="h", marker_color=BLUE, text=[f"{v/len(q):.0%}" for v in topc.values[::-1]], textposition="outside", cliponaxis=False))
        fig.update_xaxes(title="비밀번호 수(개)", range=[0, topc.max() * 1.3]); show(style(fig))
    with x3:
        gap = (q.lon - q.lof).median()
        head("온라인 vs 오프라인 크랙 시간", f"오프라인 공격은 온라인보다 중앙값 기준 약 10^{gap:.0f}배 빠릅니다. 카테고리별 중앙값입니다.")
        r = q.groupby("category")[["lon", "lof"]].median().sort_values("lon")
        fig = go.Figure()
        for i, (cat, row) in enumerate(r.iterrows()): fig.add_trace(go.Scatter(x=[row.lof, row.lon], y=[cat, cat], mode="lines", line=dict(color="#d2d2d7", width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=r.lon, y=r.index, mode="markers", name="온라인", marker=dict(size=12, color=BLUE)))
        fig.add_trace(go.Scatter(x=r.lof, y=r.index, mode="markers", name="오프라인", marker=dict(size=12, color=RED)))
        tick_axis(fig, "x"); show(style(fig))
    source("t2")

if page == 2:
    head("질문 3. 어떤 조건이 비밀번호를 강하게 만들까?", "길이와 문자 구성(대문자·숫자·특수문자)이 강도 점수와 침해 위험도에 주는 영향을 봅니다.")
    md('<span class="pill">합성 데이터 기반 탐색</span>')
    rk = st.sidebar.multiselect("침해 위험도", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    lr3 = st.sidebar.slider("길이 범위(자)", 5, 20, (5, 20), key="l3")
    h = C[C.Breach_Risk.isin(rk) & C.Password_Length.between(*lr3)]
    if h.empty: st.info("조건에 맞는 데이터가 없습니다."); st.stop()
    r1, r2 = st.columns([5, 3])
    with r1:
        kpis([("표본", len(h), 0, "개"), ("평균 강도 점수", float(h.Strength_Score.mean()), 1, "점"), ("High 위험 비율", float((h.Breach_Risk == "High").mean() * 100), 1, "%"), ("길이–점수 상관", float(h.Password_Length.corr(h.Strength_Score)) if len(h) > 2 else 0.0, 2, "")])
    with r2:
        ring([("High 위험", (h.Breach_Risk == "High").mean() * 100, "#ff6b6b", "침해 위험이 높은 비밀번호 비율"), ("Low 위험", (h.Breach_Risk == "Low").mean() * 100, "#4cc38a", "침해 위험이 낮은 비밀번호 비율")])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("길이별 평균 강도 점수", "길이가 늘수록 점수가 거의 일직선으로 오릅니다.")
        g = h.groupby("Password_Length").Strength_Score.agg(["mean", "size"]).reset_index()
        fig = go.Figure(go.Scatter(x=g.Password_Length, y=g["mean"], mode="lines+markers", fill="tozeroy", fillcolor="rgba(91,141,239,.16)", line=dict(color=BLUE, width=3), marker=dict(size=8), customdata=g["size"], hovertemplate="길이 %{x}자<br>평균 %{y:.1f}점<br>표본 %{customdata}개<extra></extra>"))
        fig.update_xaxes(title="길이(자)"); fig.update_yaxes(title="평균 강도 점수(0~100)", range=[0, 100]); show(style(fig))
    with x2:
        head("문자 구성별 평균 점수", "포함 여부로 나눈 평균입니다. 긴 비밀번호일수록 구성 요소도 많아 길이 효과가 섞여 있으니, 길이 범위를 좁혀 비교해 보세요.")
        rows = [(nm, h[h[col] > 0].Strength_Score.mean(), h[h[col] == 0].Strength_Score.mean()) for nm, col in [("대문자", "Uppercase_Count"), ("숫자", "Digit_Count"), ("특수문자", "Special_Char_Count")]]
        fig = go.Figure([go.Bar(name="포함", x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=BLUE), go.Bar(name="미포함", x=[r[0] for r in rows], y=[r[2] for r in rows], marker_color="#c7c7cc")])
        fig.update_layout(barmode="group"); fig.update_yaxes(title="평균 강도 점수", range=[0, 100]); show(style(fig))
    with x3:
        head("침해 위험도 분포", "길이 구간별 High·Medium·Low 비율입니다.")
        h = h.assign(구간=pd.cut(h.Password_Length, [0, 7, 11, 15, 20], labels=["5~7자", "8~11자", "12~15자", "16~20자"]))
        ct = pd.crosstab(h.구간, h.Breach_Risk, normalize="index") * 100
        fig = go.Figure([go.Bar(name=k, x=ct.index.astype(str), y=ct[k], marker_color=RCOL[k], text=[f"{v:.0f}%" for v in ct[k]], textposition="inside") for k in ["High", "Medium", "Low"] if k in ct])
        fig.update_layout(barmode="stack"); fig.update_yaxes(title="비율(%)", range=[0, 100]); show(style(fig))
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

if page == 3:
    head("질문 4. 내 비밀번호는 안전할까?", "입력한 비밀번호는 저장하지 않으며, 유출 확인에는 해시 앞 5글자만 전송합니다(k-익명성).")
    with st.expander("데이터 출처 · 계산 기준 · 한계"):
        st.markdown("**유출 확인**: Have I Been Pwned의 Pwned Passwords range API(`api.pwnedpasswords.com/range`, 키 불필요)를 사용합니다. SHA-1 해시의 앞 5글자만 보내고 나머지는 브라우저가 아닌 앱 서버에서 대조하며, 비밀번호 원문은 저장하지 않습니다. 계정(이메일) 유출 조회용 v3 API는 유료 키가 필요해 사용하지 않았습니다.\n\n**점수**: 길이×log2(문자 종류 수)로 추정한 엔트로피에 패턴 감점을 적용해 80bit를 100점으로 환산했고, 유출·흔한 비밀번호는 상한을 둡니다. 교육용 근사치이며 실제 안전성을 보증하지 않습니다.")
    _, right = st.columns([1, 2], gap="large")
    with right:
        st.caption("예시로 빠르게 확인")
        examples = st.columns(3, gap="small")
        for col, (label, value) in zip(examples, [("흔한 비밀번호", "password123"), ("규칙형", "Summer2024!"), ("긴 무작위", "vK9#mQ2$zL7@pW4xR8")]):
            if col.button(label, use_container_width=True): diagnose(value, True)
        with st.form("f", clear_on_submit=True):
            pw = st.text_input("비밀번호 입력", type="password", placeholder="실제 사용 중인 비밀번호는 발표 중 입력하지 마세요")
            u = st.checkbox("유출 여부 확인(Have I Been Pwned)", value=True)
            if st.form_submit_button("진단하기", use_container_width=True) and pw: diagnose(pw, u)
    d = st.session_state.get("diag")
    if d:
        s = d["score"]; grade, col = ("위험", RED) if s < 30 else ("주의", ORANGE) if s < 55 else ("양호", BLUE) if s < 80 else ("안전", GREEN)
        if d["cnt"] and d["cnt"] > 0: grade, col = "위험", RED
        c1, c2, c3 = st.columns([2, 3, 3])
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(s), number=dict(suffix="점", font=dict(size=44)), title=dict(text=f"위험도 등급: {grade}", font=dict(size=18)),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.35), bgcolor="#f2f2f5", borderwidth=0, steps=[dict(range=[0, 30], color="#ffe9e7"), dict(range=[30, 55], color="#fff2dc"), dict(range=[55, 80], color="#e5f1ff"), dict(range=[80, 100], color="#e3f8e9")])))
            show(style(fig, 280))
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
