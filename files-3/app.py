import hashlib, html as _html, inspect, math, re, time
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, requests, streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

st.set_page_config(page_title="비밀번호는 얼마나 버틸까?", page_icon="🔐", layout="wide", initial_sidebar_state="collapsed")
D = Path(__file__).parent / "data"
FONT = "Pretendard, -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
BLUE, GREEN, ORANGE, RED, GRAY, YEL = "#5b8def", "#4cc38a", "#ff9f0a", "#ff3b30", "#86868b", "#ffcc00"
KO = {"Very week": "매우 약함", "Week": "약함", "Average": "보통", "Strong": "강함", "Very strong": "매우 강함"}
ORDER = list(KO.values()); CCOL = dict(zip(ORDER, [RED, ORANGE, YEL, BLUE, GREEN])); RCOL = {"High": RED, "Medium": ORANGE, "Low": GREEN}
YR = 31557600
CAND = [1, 60, 3600, 86400, YR, YR * 1e2, YR * 1e4, YR * 1e6, YR * 1e9, YR * 1e12, YR * 1e18, YR * 1e24, YR * 1e36, YR * 1e60]
SRC = {"t1": ("Password Metrics Dataset", "원본 100,000행 중 중복 비밀번호를 제거한 고유 25,000개(원문은 저장하지 않음). 강도 등급은 등급별 5,000개씩 균등 표본이라 실제 이용자 비율이 아님. 뚫리는 시간은 10^-7~10^104초로 값 범위가 넓어 로그 눈금·중앙값으로 집계. 'Very week' 오타는 '매우 약함'으로 정리.", ""),
       "t2": ("Password Strength and Vulnerability Dataset", "흔한 비밀번호 500개(인기 순위 기준). Kaggle 설명에 따르면 원본은 Information is Beautiful의 자료입니다. value·time_unit은 '온라인 추측으로 뚫는 데 걸리는 시간', offline_crack_sec는 '오프라인으로 뚫는 시간(초)'이며, 둘을 같은 초 단위로 환산(1개월=30.4일, 1년=365.25일)해 비교했습니다. strength 열은 이 목록 안의 상대 점수라 쓰지 않았습니다.", "https://www.kaggle.com/datasets/utkarshx27/passwords")}

def md(s): st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)
def human(s):
    if s is None or not np.isfinite(s): return "측정 불가"
    if s < 1e-3: return "즉시"
    if s < 1: return f"{s*1000:.0f}밀리초"
    for lim, div, u in [(60, 1, "초"), (3600, 60, "분"), (86400, 3600, "시간"), (2629800, 86400, "일"), (YR, 2629800, "개월")]:
        if s < lim: return f"{s/div:.0f}{u}"
    y = s / YR
    return f"{y:,.0f}년" if y < 1e4 else "10" + str(int(math.log10(y))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")) + "년"
def tick_axis(fig, lo, hi, axis="y"):
    lo, hi = float(lo), float(hi); span = max(hi - lo, 1.0); keep = []
    for v in CAND:
        x = math.log10(v)
        if lo - .3 <= x <= hi + .3 and (not keep or x - keep[-1] >= span / 6): keep.append(x)
    kw = dict(tickvals=keep, ticktext=[human(10 ** x) for x in keep], range=[lo - .4, hi + .4], tickangle=0)
    if axis == "y":
        fig.update_yaxes(**kw)
    else:
        fig.update_xaxes(**kw)
def style(fig, h=310):
    fig.update_layout(height=h, margin=dict(l=12, r=16, t=26, b=14), paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(family=FONT, size=12, color="#1d1d1f"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)), hoverlabel=dict(font_family=FONT, bgcolor="white"))
    fig.update_xaxes(gridcolor="#ececf0", zeroline=False, showline=False, automargin=True); fig.update_yaxes(gridcolor="#ececf0", zeroline=False, showline=False, automargin=True)
    return fig
_PW = "width" in inspect.signature(st.plotly_chart).parameters
def show(fig, key):
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key, **(dict(width="stretch") if _PW else dict(use_container_width=True)))
_bd = [0]
def head(title, take):
    b = ""
    if title.startswith("질문"): _bd[0] = 0
    else: _bd[0] += 1; b = f'<span class="bd">분석 {_bd[0]}</span>'
    md(f'<div class="sec"><div class="t">{b}{title}</div><div class="k">{take}</div></div>')
def source(key):
    n, t, u = SRC[key]
    with st.expander("데이터 출처 · 집계 기준 · 한계"): st.markdown(f"**{n}**\n\n{t}" + (f"\n\n🔗 [원본 데이터 보기(Kaggle)]({u})" if u else ""))
def ins(text, color=BLUE, label="핵심 결론"): md(f'<div class="ins" style="border-left-color:{color}"><b>{label}</b> {text}</div>')
def tile_grid(qq):
    lo, hi = qq.lon.min(), qq.lon.max(); A, B, Cc = (255, 154, 143), (255, 228, 154), (143, 220, 176)
    def col(v):
        t = 0.0 if hi == lo else (v - lo) / (hi - lo); p, q_, u = (A, B, t * 2) if t < .5 else (B, Cc, (t - .5) * 2)
        return "rgb(%d,%d,%d)" % tuple(int(p[k] + (q_[k] - p[k]) * u) for k in range(3))
    cells = "".join(f'<div class="tl" style="background:{col(v)};animation-delay:{i*35}ms" title="{_html.escape(str(w))} · {human(s_)}"><span class="rk">{int(r)}위</span><b>{_html.escape(str(w))}</b><span class="tm">{human(s_)}</span></div>' for i, (r, w, v, s_) in enumerate(zip(qq["rank"], qq.password, qq.lon, qq.online_sec)))
    md(f'<div class="tg">{cells}</div>')
def kpis(items):
    cards = "".join(f'<div class="c" style="animation-delay:{i*80}ms"><div class="l">{l}</div><div class="v" style="font-size:{26 if isinstance(v,(int,float)) else 18}px"><span class="n" data-v="{v if isinstance(v,(int,float)) else ""}" data-d="{d}">{v if not isinstance(v,(int,float)) else f"{v:,.{d}f}"}</span><span class="u">{u}</span></div></div>' for i, (l, v, d, u) in enumerate(items))
    h = f"""<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css" rel="stylesheet"><style>
    body{{margin:0;padding:4px 2px 10px;font-family:{FONT}}}.g{{display:grid;grid-template-columns:repeat({len(items)},minmax(0,1fr));gap:12px}}
    .c{{background:#fff;border-radius:18px;padding:14px 16px 16px;border-left:5px solid #4cc38a;box-shadow:0 2px 12px rgba(0,0,0,.06);animation:u .6s cubic-bezier(.2,.8,.2,1) both;min-width:0}}
    .l{{color:#6e6e73;font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .v{{margin-top:6px;color:#1d1d1f;font-weight:700;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.25}}.u{{font-size:14px;color:#6e6e73;margin-left:3px}}
    @keyframes u{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}</style>
    <div class="g">{cards}</div><script>document.querySelectorAll('.n').forEach(e=>{{const t=parseFloat(e.dataset.v);if(isNaN(t))return;const d=+e.dataset.d,s=performance.now();
    const f=n=>{{const p=Math.min((n-s)/1100,1),x=1-Math.pow(1-p,3);e.textContent=(t*x).toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}});if(p<1)requestAnimationFrame(f)}};requestAnimationFrame(f)}})</script>"""
    try: components.html(h, height=116)
    except Exception: md("".join(f"<b>{l}</b> {v}{u} &nbsp; " for l, v, d, u in items))

@st.cache_data
def load():
    m = pd.read_csv(D / "metrics_unique.csv"); m["등급"] = pd.Categorical(m.class_strength.map(KO), ORDER, ordered=True); m["lc"] = np.log10(m.crack_time_sec.clip(lower=1e-12))
    p = pd.read_csv(D / "common_passwords.csv"); p["lon"] = np.log10(p.online_sec); p["lof"] = np.log10(p.offline_crack_sec)
    return m, p
M, P = load(); TOP = set(P.password.str.lower())

md("""<style>@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@property --p{syntax:'<integer>';inherits:true;initial-value:0}
html,body,[class*="css"],.stApp{font-family:Pretendard,-apple-system,'Apple SD Gothic Neo',sans-serif!important}
.stApp{background:#e9edf2}#MainMenu,footer{visibility:hidden}[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
.block-container{max-width:1600px;padding:1.2rem 1.6rem 1.5rem;position:relative;z-index:1;animation:up .6s both}[data-testid="stVerticalBlock"]{gap:.6rem}
.stApp::before,.stApp::after{content:"";position:fixed;width:560px;height:560px;border-radius:50%;z-index:0;pointer-events:none;filter:blur(30px)}
.stApp::before{top:-160px;right:-100px;background:radial-gradient(circle,#a9c8ff99,transparent 70%);animation:fl 13s ease-in-out infinite alternate}
.stApp::after{bottom:-200px;left:200px;background:radial-gradient(circle,#9fe8c299,transparent 70%);animation:fl 17s ease-in-out infinite alternate-reverse}
@keyframes fl{to{transform:translate(-80px,60px) scale(1.2)}}@keyframes up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes wipe{from{opacity:0;clip-path:inset(0 100% 0 0 round 22px)}to{opacity:1;clip-path:inset(0 0 0 0 round 22px)}}
.st-key-hdr{background:linear-gradient(100deg,#171f3d,#0f1730);border-radius:22px;padding:14px 22px;border:1px solid #25305a;box-shadow:0 6px 24px rgba(15,23,48,.25)}
.st-key-hdr *{color:#e6e9f2}.ht{font-size:19px;font-weight:800;letter-spacing:-.02em;white-space:nowrap}.ht small{display:block;font-size:11.5px;color:#8d97b5!important;font-weight:500;margin-top:3px}
.st-key-hdr [data-testid="stButtonGroup"]{flex-wrap:nowrap;gap:8px}
.st-key-hdr [data-testid="stButtonGroup"] button{min-height:50px;padding:10px 22px;font-size:16px;font-weight:700;border-radius:15px;background:#1a2547;border:1px solid #2a3868;white-space:nowrap;transition:all .25s}
.st-key-hdr [data-testid="stButtonGroup"] button:hover{transform:translateY(-2px);border-color:#5b8def}
.st-key-hdr [data-testid="stButtonGroup"] button[aria-checked="true"],.st-key-hdr [data-testid="stButtonGroup"] button[aria-pressed="true"],.st-key-hdr button[data-testid="stBaseButton-segmented_controlActive"]{background:#5b8def!important;border-color:#8fb4ff;box-shadow:0 0 20px rgba(91,141,239,.6)}
.st-key-hdr [role="radiogroup"]{flex-direction:row;gap:8px;flex-wrap:nowrap}.st-key-hdr [role="radiogroup"] label{padding:10px 18px;border-radius:14px;background:#1a2547;border:1px solid #2a3868}
.st-key-hdr button[kind="secondary"]{background:#1a2547;border:1px solid #2a3868;border-radius:12px}
.sec{padding:4px 2px 8px;min-height:86px}.sec .t{font-size:16.5px;font-weight:800;letter-spacing:-.02em;line-height:1.35}.sec .k{font-size:12.8px;color:#5b6472;margin-top:4px;line-height:1.5}
.bd{background:#5b8def;color:#fff;font-size:11px;font-weight:700;border-radius:6px;padding:2px 8px;margin-right:8px;vertical-align:2px}
div[data-testid="stPlotlyChart"]{background:#fff;border-radius:22px;overflow:hidden;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:wipe 1.1s cubic-bezier(.2,.8,.2,1) both;transition:box-shadow .3s}
div[data-testid="stPlotlyChart"]:hover{box-shadow:0 14px 34px rgba(60,90,140,.16)}
.rw{position:relative;width:80px;height:80px;flex:none;counter-reset:n var(--p);animation:fill 1.8s cubic-bezier(.2,.8,.2,1) .3s forwards}
.ins{background:#fff;border-left:5px solid #5b8def;border-radius:14px;padding:12px 18px;font-size:14px;line-height:1.6;box-shadow:0 2px 12px rgba(0,0,0,.05);animation:up .6s both}.ins b{margin-right:8px}
.box{background:#fff;border-radius:22px;padding:18px 22px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .6s both;line-height:1.7;font-size:14px}.box b{font-size:17px}
.chip{display:inline-block;border-radius:99px;padding:5px 12px;margin:3px 4px 3px 0;font-size:13.5px;font-weight:600}.ok{background:#e3f8e9;color:#1b7a36}.no{background:#ffe9e7;color:#c4261b}
span[data-baseweb="tag"]{background:#5b8def!important;border-radius:8px}[role="slider"]{background:#5b8def!important}[data-testid="stExpander"]{border-radius:14px;background:#fff}
.spl{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.9s ease forwards}
.spl div{color:#fff;font-size:clamp(30px,6vw,64px);font-weight:800;letter-spacing:-.03em;animation:si 1.6s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.92);filter:blur(10px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,72%{opacity:1}100%{opacity:0;visibility:hidden}}
.tg{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;background:#fff;border-radius:22px;padding:14px;box-shadow:0 2px 14px rgba(0,0,0,.05);min-height:300px;box-sizing:border-box;align-content:start}.tl{border-radius:14px;padding:10px 6px;text-align:center;min-width:0;display:flex;flex-direction:column;gap:3px;justify-content:center;animation:up .5s both;transition:transform .25s}.tl:hover{transform:translateY(-3px) scale(1.03)}.tl b{font-size:14px;color:#1d1d1f;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rk{font-size:11px;color:rgba(0,0,0,.55);font-weight:600}.tm{font-size:12px;font-weight:700;color:rgba(0,0,0,.75);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media(max-height:960px){.block-container{zoom:.9}}@media(max-height:880px){.block-container{zoom:.82}}@media(max-height:800px){.block-container{zoom:.74}}@media(max-height:720px){.block-container{zoom:.66}}@media(max-height:640px){.block-container{zoom:.58}}
@media(prefers-reduced-motion:reduce){*{animation-duration:.01s!important}}</style>""")
if not st.session_state.get("splash"):
    st.session_state["splash"] = True; md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>')

NAMES = ["👋 시작하기", "📊 뚫리는 시간", "⚠️ 흔한 비밀번호", "🔍 내 비밀번호 점검"]
if "nav" not in st.session_state: st.session_state["nav"] = NAMES[0]
def _keep():
    if st.session_state.get("nav") is None: st.session_state["nav"] = st.session_state.get("last_nav", NAMES[0])
@st.fragment(run_every=12)
def tour():
    if st.session_state.get("tour_on") and time.time() - st.session_state.get("tour_t", 0) > 11:
        st.session_state["nav"] = NAMES[(NAMES.index(st.session_state["nav"]) + 1) % len(NAMES)]; st.session_state["tour_t"] = time.time(); st.rerun()
with st.container(key="hdr"):
    hc = st.columns([2.3, 7, 1.3, 1.2], vertical_alignment="center")
    hc[0].markdown('<div class="ht">🔐 비밀번호 보안 분석<small>얼마나 빨리 뚫리고, 어떻게 지킬까</small></div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"): hc[1].segmented_control("메뉴", NAMES, key="nav", label_visibility="collapsed", on_change=_keep)
    else: hc[1].radio("메뉴", NAMES, horizontal=True, label_visibility="collapsed", key="nav")
    hc[2].toggle("자동 순회", key="tour_on", on_change=lambda: st.session_state.update(tour_t=time.time()))
    hc[3].button("▶ 인트로", on_click=lambda: st.session_state.update(splash=False))
tour()
cur = st.session_state.get("nav") or NAMES[0]; st.session_state["last_nav"] = cur; page = NAMES.index(cur)

def filters(n=3): return st.columns([2, 2, 3][:n], vertical_alignment="center")

def p_intro():
    head("질문 0. 비밀번호가 '뚫린다'는 건 무슨 뜻일까?", "처음 보시는 분을 위한 요약판입니다. 아래 기준을 바꾸면 숫자와 차트가 함께 바뀝니다.")
    fl = filters(); TS = {"1분": 60, "1시간": 3600, "1일": 86400, "1년": YR, "100년": YR * 100}
    tk = fl[0].select_slider("이 시간 안에 뚫리면 '위험'으로 셉니다", options=list(TS), value="1년", key="i_t"); T = TS[tk]
    Ln = fl[1].slider("내 비밀번호 길이(자)", 4, 30, 8, key="i_l"); fl[2].caption("↻ 기준을 바꾸면 아래 숫자와 차트가 바뀝니다")
    pc = [(M.crack_time_sec <= T).mean() * 100, (P.online_sec <= T).mean() * 100, (P.offline_crack_sec <= T).mean() * 100]
    ins(f"해킹 프로그램은 비밀번호를 하나씩 넣어 보며 맞는 것을 찾습니다. 이 대시보드의 <b>뚫리는 시간</b>은 그 프로그램이 맞힐 때까지 걸리는 시간이고 길수록 안전합니다(숫자 4자리는 가능한 조합이 1만 개뿐). 같은 기준({tk})으로 보면 유출 표본의 {pc[0]:.0f}%, 흔한 비밀번호는 온라인 {pc[1]:.0f}%·오프라인 {pc[2]:.0f}%가 뚫립니다. 어디서 공격받느냐에 따라 위험이 크게 달라집니다.", GREEN, "한눈에 보기")
    kpis([("분석한 비밀번호", len(M), 0, "개"), ("흔한 비밀번호 목록", len(P), 0, "개"), (f"{tk} 안에 뚫림 · 유출 표본", pc[0], 0, "%"), (f"{tk} 안에 뚫림 · 흔한 것(온라인)", pc[1], 0, "%"), (f"{tk} 안에 뚫림 · 흔한 것(오프라인)", pc[2], 0, "%")])
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("100개 중 몇 개가 뚫릴까?(와플 차트)", f"네모 한 칸이 비밀번호 100개 중 1개, 빨강은 {tk} 안에 뚫리는 것입니다. 유출 표본은 등급별 균등 추출이라 실제 비율은 아닙니다.")
        ttl = ["유출 표본", "흔한·온라인", "흔한·오프라인"]; fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t}<br><b>{p:.0f}%</b>" for t, p in zip(ttl, pc)])
        for i, p in enumerate(pc):
            z = np.zeros(100); z[:int(round(p))] = 1
            fig.add_trace(go.Heatmap(z=z.reshape(10, 10), colorscale=[[0, "#cfeedd"], [1, "#ff6b6b"]], showscale=False, xgap=3, ygap=3, hoverinfo="skip"), 1, i + 1)
            fig.update_xaxes(visible=False, row=1, col=i + 1); fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i + 1)
        fig.update_annotations(font_size=11); show(style(fig, 300).update_layout(margin=dict(l=12, r=12, t=64, b=12)), "i1")
    with x2:
        head("내 비밀번호 길이면 얼마나 버틸까?(막대 게이지)", "슬라이더로 고른 길이 ±1자 비밀번호의 중간값입니다. 빨강은 하루 안, 주황은 1년 안, 초록은 1년 넘게 버티는 구간입니다.")
        sub = M[M.length.between(Ln - 1, Ln + 1)]; m = sub.crack_time_sec.median(); lm = float(np.clip(math.log10(max(m, 1e-6)), 0, 14))
        md(f'<div style="text-align:center;font-size:32px;font-weight:800;letter-spacing:-.02em;padding:6px 0">{human(m)}</div>')
        tv = [0, 3.556, 4.936, 7.499, 9.499, 13.499]
        fig = go.Figure(go.Indicator(mode="gauge", value=lm, gauge=dict(shape="bullet", axis=dict(range=[0, 14], tickvals=tv, ticktext=["1초", "1시간", "1일", "1년", "100년", "100만년"]), bar=dict(color="#1d1d1f", thickness=.45),
            steps=[dict(range=[0, 4.936], color="#ffd9d6"), dict(range=[4.936, 7.499], color="#ffe9c7"), dict(range=[7.499, 14], color="#d5f3e1")])))
        fig.update_layout(margin=dict(l=40, r=40, t=30, b=20)); show(style(fig, 200).update_layout(margin=dict(l=40, r=40, t=30, b=20)), "i2")
        st.markdown(f"길이 **{Ln}자** 안팎의 비밀번호는 보통 **{human(m)}**이 걸려야 뚫립니다. (표본 {len(sub):,}개)")
    with x3:
        head("길이 → 강도 등급 흐름(산키 차트)", "선이 굵을수록 비밀번호가 많습니다. 길이 구간별로 어떤 강도 등급이 되는지 따라가 보세요.")
        bins = ["≤7자", "8~11자", "12~15자", "16~20자", "21자+"]
        sk = M.assign(구간=pd.cut(M.length, [0, 7, 11, 15, 20, 70], labels=bins).astype(str), 등급=M.등급.astype(str)).groupby(["구간", "등급"]).size().reset_index(name="n")
        fig = go.Figure(go.Sankey(node=dict(label=bins + ORDER, color=["#9db7ee"] * 5 + [CCOL[o] for o in ORDER], pad=14, thickness=14), link=dict(source=[bins.index(a) for a in sk.구간], target=[5 + ORDER.index(b) for b in sk.등급], value=sk.n, color="rgba(91,141,239,.22)")))
        show(style(fig, 300), "i3")
    with st.expander("데이터 출처 · 지도(지역) 시각화를 넣지 않은 이유"):
        st.markdown("두 데이터 모두 국가·지역 정보가 없어 지도 시각화는 넣지 않았습니다. 데이터에 없는 위치를 임의로 만들어 붙이는 것은 정직한 시각화가 아니라고 판단했습니다. 데이터별 출처와 집계 기준은 각 페이지 하단에 있습니다.\n\n🔗 흔한 비밀번호 목록: [Kaggle에서 보기](https://www.kaggle.com/datasets/utkarshx27/passwords)")

def p_leak():
    head("질문 1. 비밀번호가 길수록 얼마나 오래 버틸까?", "고유 비밀번호 25,000개를 등급과 길이별로 살펴봅니다. 시간은 '해킹 프로그램이 맞힐 때까지 걸리는 시간'입니다.")
    fl = filters(); gs = fl[0].multiselect("강도 등급", ORDER, default=ORDER, key="a_g"); lr = fl[1].slider("길이 범위(자)", 4, 63, (4, 40), key="a_l"); fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")
    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty: st.info("조건에 맞는 데이터가 없습니다. 등급이나 길이 범위를 넓혀 보세요."); return
    kpis([("고유 비밀번호", len(f), 0, "개"), ("길이 중앙값", float(f.length.median()), 0, "자"), ("예측 어려움 점수 중앙값", float(f.entropy.median()), 1, "bit"), ("뚫리는 시간 중앙값", human(f.crack_time_sec.median()), 0, ""), ("1년 안에 뚫림", (f.crack_time_sec <= YR).mean() * 100, 0, "%"), ("1시간 안에 뚫림", (f.crack_time_sec <= 3600).mean() * 100, 0, "%")])
    g = f.groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=20").reset_index(); slope = np.polyfit(f.length, f.lc, 1)[0] if f.length.nunique() > 1 else float("nan")
    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        head("길이가 길수록 얼마나 오래 걸릴까?(선 그래프)", (f"길이가 1자 늘 때마다 뚫리는 시간이 평균 약 {10**slope:.1f}배 늘어납니다. 세로 눈금은 10배씩 커지는 로그이고, 표본 20개 미만인 길이는 뺐습니다." if np.isfinite(slope) else "길이 구간을 넓히면 추세가 보입니다."))
        if g.empty: st.info("표본이 부족합니다.")
        else:
            y = np.log10(g.med); fig = go.Figure(go.Scatter(x=g.length, y=y, mode="lines+markers", line=dict(color=BLUE, width=3, shape="spline"), marker=dict(size=7), customdata=[[human(v), n] for v, n in zip(g.med, g.n)], hovertemplate="길이 %{x}자<br>중앙값 %{customdata[0]}<br>표본 %{customdata[1]}개<extra></extra>"))
            fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=GRAY, annotation_text="1년", annotation_position="top left"); fig.update_xaxes(title="비밀번호 길이(자)"); tick_axis(fig, min(y.min(), math.log10(YR)), y.max()); show(style(fig), "a1")
            with st.popover("📋 표 보기"): st.dataframe(g.assign(중앙값=g.med.map(human))[["length", "n", "중앙값"]].rename(columns={"length": "길이", "n": "표본"}), hide_index=True)
    with x2:
        stp = [("시작", 0), ("1분 버팀", 60), ("1시간 버팀", 3600), ("1일 버팀", 86400), ("1년 버팀", YR), ("100년 버팀", YR * 100)]
        vals = [100.0] + [(f.crack_time_sec > t).mean() * 100 for _, t in stp[1:]]
        head("시간이 지날수록 몇 %가 버틸까?(깔때기 차트)", f"'버틴다'는 아직 안 뚫렸다는 뜻입니다. 선택한 비밀번호 중 {vals[3]:.0f}%가 1일을, {vals[4]:.0f}%가 1년을 버팁니다.")
        fig = go.Figure(go.Funnel(y=[a for a, _ in stp], x=vals, text=[f"{v:.0f}%" for v in vals], textinfo="text", textposition="inside", textfont=dict(size=13, color="#fff"), marker=dict(color=["#5b8def", "#6e9bf0", "#4cc38a", "#3bb27a", "#2a9d68", "#1f8a58"]), connector=dict(line=dict(color="#e5e7eb"))))
        show(style(fig), "a2")
    with x3:
        head("예측 어려움 점수와 뚫리는 시간(산점도)", f"예측 어려움 점수(엔트로피)는 값이 클수록 추측하기 어렵다는 뜻입니다. 둘의 상관은 r={np.corrcoef(f.entropy, f.lc)[0,1]:.2f}로, 점수가 길이에서 거의 그대로 계산돼 점들이 한 줄로 늘어섭니다." if len(f) > 2 else "")
        sm = f.sample(min(2000, len(f)), random_state=7); fig = go.Figure([go.Scatter(x=d.entropy, y=d.lc, mode="markers", name=k, marker=dict(size=5, color=CCOL[k], opacity=.6)) for k, d in sm.groupby("등급", observed=True)])
        fig.update_xaxes(title="예측 어려움 점수(bit)"); tick_axis(fig, sm.lc.min(), sm.lc.max()); show(style(fig), "a3")
    ins(f"선택한 {len(f):,}개 중 {(f.crack_time_sec <= YR).mean():.0%}가 1년 안에 뚫립니다. 길이 중앙값 {int(f.length.median())}자 비밀번호의 뚫리는 시간 중앙값은 {human(f.crack_time_sec.median())}입니다.")
    source("t1")

def p_common():
    head("질문 2. 흔한 비밀번호는 얼마나 빨리 뚫릴까?", "온라인 = 로그인 창에 직접 입력해 보는 공격(느림), 오프라인 = 유출된 파일을 컴퓨터로 대량 대입하는 공격(매우 빠름)입니다.")
    fl = filters(); n = fl[0].slider("상위 몇 개를 볼까요?", 5, 100, 20, 5, key="b_n"); allc = sorted(P.category.unique()); cats = fl[1].multiselect("비밀번호 종류", allc, default=allc, key="b_c"); fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")
    q = P[(P["rank"] <= n) & P.category.isin(cats)]
    if q.empty: st.info("조건에 맞는 비밀번호가 없습니다. 종류를 선택해 주세요."); return
    topc = q.category.value_counts()
    kpis([("표시 중 비밀번호", len(q), 0, "개"), ("가장 많은 종류", f"{topc.index[0]}", 0, f" {topc.iloc[0]/len(q):.0%}"), ("온라인 공격 · 중앙값", human(q.online_sec.median()), 0, ""), ("오프라인 공격 · 중앙값", human(q.offline_crack_sec.median()), 0, ""), ("온라인 · 1일 안에 뚫림", (q.online_sec <= 86400).mean() * 100, 0, "%"), ("오프라인 · 1초 안에 뚫림", (q.offline_crack_sec <= 1).mean() * 100, 0, "%")])
    x1, x2, x3 = st.columns([4, 3, 3]); qq = q.head(20)
    with x1:
        head(f"가장 흔한 비밀번호 TOP {len(qq)}, 얼마나 빨리 뚫릴까?(타일 지도)", "칸 하나가 비밀번호 하나입니다. 빨강일수록 빨리 뚫리고(온라인 공격 기준) 초록일수록 오래 걸립니다." + (" 상위 20개만 보여 줍니다." if len(q) > 20 else ""))
        tile_grid(qq)
        with st.popover("📋 표 보기"): st.dataframe(qq.assign(온라인=qq.online_sec.map(human), 오프라인=qq.offline_crack_sec.map(human))[["rank", "password", "category", "온라인", "오프라인"]].rename(columns={"rank": "순위", "password": "비밀번호", "category": "종류"}), hide_index=True)
    with x2:
        head("어떤 종류가 가장 많을까?(도넛 차트)", f"가장 많은 종류는 '{topc.index[0]}'({topc.iloc[0]}개)입니다. 조각이 클수록 그 종류가 많습니다.")
        fig = go.Figure(go.Pie(labels=topc.index, values=topc.values, hole=.58, sort=False, textinfo="percent", marker=dict(colors=px.colors.qualitative.Set3 * 2, line=dict(color="#fff", width=2))))
        fig.add_annotation(text=f"<b>{len(q)}개</b>", x=.5, y=.5, showarrow=False, font=dict(size=22)); show(style(fig, 300).update_layout(legend=dict(orientation="v", x=1.0, y=.5, font=dict(size=11))), "b2")
    with x3:
        a_, b_ = (q.online_sec <= 86400).mean() * 100, (q.offline_crack_sec <= 86400).mean() * 100
        head("시간이 지날수록 몇 %가 뚫릴까?(누적 면적 차트)", f"가로는 걸리는 시간, 세로는 그 시간 안에 뚫린 비율입니다. 하루 안에 온라인 공격으로 {a_:.0f}%, 오프라인 공격으로 {b_:.0f}%가 뚫립니다. 선이 왼쪽에 있을수록 빨리 뚫립니다.")
        fig = go.Figure()
        for nm, col, rgba, v in [("온라인(로그인 창)", BLUE, "rgba(91,141,239,.18)", q.lon), ("오프라인(유출 파일)", RED, "rgba(255,59,48,.14)", q.lof)]:
            xs = np.sort(v.values); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
            fig.add_trace(go.Scatter(x=xs, y=ys, name=nm, mode="lines", line=dict(color=col, width=3, shape="hv"), fill="tozeroy", fillcolor=rgba, hovertemplate="%{y:.0f}%가 뚫림<extra>" + nm + "</extra>"))
        tick_axis(fig, min(q.lon.min(), q.lof.min()), max(q.lon.max(), q.lof.max()), "x"); fig.update_yaxes(title="뚫린 비율(%)", range=[0, 102]); show(style(fig), "b3")
    ins(f"표시 중 {len(q)}개 중 {(q.online_sec <= 86400).mean():.0%}는 로그인 창 공격에도 하루 안에 뚫리고, 유출 파일을 대입하는 공격은 중앙값 {human(q.offline_crack_sec.median())}이면 충분합니다.")
    source("t2")

def pool_size(p): return 26 * bool(re.search("[a-z]", p)) + 26 * bool(re.search("[A-Z]", p)) + 10 * bool(re.search("[0-9]", p)) + 33 * bool(re.search(r"[ -/:-@\[-`{-~]", p)) + 100 * bool(re.search(r"[^\x00-\x7f]", p))
SEQ = ["abcdefghijklmnopqrstuvwxyz", "0123456789", "qwertyuiopasdfghjklzxcvbnm"]
@st.cache_data(ttl=3600, show_spinner=False)
def hibp_range(prefix):
    r = requests.get(f"https://api.pwnedpasswords.com/range/{prefix}", headers={"Add-Padding": "true", "User-Agent": "streamlit-password-dashboard"}, timeout=6); r.raise_for_status(); return r.text
def diagnose(pw, use_hibp):
    L = len(pw); low = pw.lower(); pool = pool_size(pw); ent = L * math.log2(pool) if pool else 0
    seq = any(s[i:i + 3] in low or s[i:i + 3][::-1] in low for s in SEQ for i in range(len(s) - 2)); rep = bool(re.search(r"(.)\1\1", pw)); yr = bool(re.search(r"(19|20)\d\d", pw)); ent -= 15 * seq + 10 * rep + 8 * yr; cnt = None
    if use_hibp:
        try:
            sha = hashlib.sha1(pw.encode()).hexdigest().upper()
            cnt = next((int(c) for h_, _, c in (x.partition(":") for x in hibp_range(sha[:5]).splitlines()) if h_ == sha[5:]), 0)
        except Exception: cnt = -1
    score = max(0, min(100, ent / 80 * 100)); common = low in TOP
    if common: score = min(score, 5)
    if cnt and cnt > 0: score = min(score, 10)
    st.session_state["diag"] = dict(L=L, score=score, ent=max(ent, 0), cnt=cnt, common=common, seq=seq, rep=rep, yr=yr, up=bool(re.search("[A-Z]", pw)), dg=bool(re.search("[0-9]", pw)), sp=bool(re.search(r"[^A-Za-z0-9]", pw)), allnum=pw.isdigit())

def p_check():
    head("질문 4. 내 비밀번호는 안전할까?", "입력한 비밀번호는 저장하지 않습니다. 유출 확인에는 암호화한 값의 앞 5글자만 보냅니다.")
    ex = st.columns(3); u = True
    for i, (col, (lab, v)) in enumerate(zip(ex[:3], [("예시 · 흔한 비밀번호", "password123"), ("예시 · 규칙형", "Summer2024!"), ("예시 · 긴 무작위", "vK9#mQ2$zL7@pW4xR8")])):
        if col.button(lab, key=f"d_ex{i}"): diagnose(v, u)
    with st.form("f", clear_on_submit=True, border=False):
        fc = st.columns([6, 1]); pw = fc[0].text_input("비밀번호 입력", type="password", placeholder="비밀번호를 입력하세요(저장되지 않습니다)", label_visibility="collapsed")
        if fc[1].form_submit_button("점검하기") and pw: diagnose(pw, u)
    d = st.session_state.get("diag")
    if d:
        s = d["score"]; grade, col = ("위험", RED) if s < 30 else ("주의", ORANGE) if s < 55 else ("양호", BLUE) if s < 80 else ("안전", GREEN)
        if d["cnt"] and d["cnt"] > 0: grade, col = "위험", RED
        c1, c2, c3 = st.columns([2, 3, 3])
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(s), number=dict(suffix="점", font=dict(size=40)), title=dict(text=f"등급: {grade}", font=dict(size=17)), gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.35), bgcolor="#f2f2f5", borderwidth=0, steps=[dict(range=[0, 30], color="#ffe9e7"), dict(range=[30, 55], color="#fff2dc"), dict(range=[55, 80], color="#e5f1ff"), dict(range=[80, 100], color="#e3f8e9")])))
            show(style(fig, 230), "d1")
            vf = go.Figure([go.Violin(x=M[M.등급 == g_].length, name=g_, orientation="h", line_color=CCOL[g_], points=False, showlegend=False, spanmode="hard") for g_ in ORDER]); vf.add_vline(x=d["L"], line_color="#1d1d1f", line_dash="dash", annotation_text="내 길이")
            vf.update_xaxes(title="분석한 비밀번호의 등급별 길이 분포(자)"); show(style(vf, 230), "d2")
        with c2:
            chips = [("8자 이상", d["L"] >= 8), ("12자 이상", d["L"] >= 12), ("대문자", d["up"]), ("숫자", d["dg"]), ("특수문자", d["sp"]), ("흔한 목록에 없음", not d["common"]), ("연속 문자 없음", not d["seq"]), ("반복 문자 없음", not d["rep"]), ("연도 패턴 없음", not d["yr"])]
            ch = "".join(f'<span class="chip {"ok" if v else "no"}">{"✓" if v else "✕"} {k}</span>' for k, v in chips)
            br = {None: "유출 확인을 사용하지 않았습니다.", -1: "유출 확인 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요."}.get(d["cnt"], f"유출 기록 <b>{d['cnt']:,}회</b> 발견. 이미 공개된 비밀번호이니 바로 바꾸세요." if (d["cnt"] or 0) > 0 else "유출 기록에서 발견되지 않았습니다(안전하다는 뜻은 아닙니다).")
            pct = (M.length < d["L"]).mean()
            md(f'<div class="box"><b>점검 결과</b><br>{ch}<br>예측 어려움 점수 약 {d["ent"]:.0f}bit · 유출 파일을 대입하는 공격(초당 100억 회 가정)이면 약 {human(2**d["ent"]/2/1e10)}<br>{br}<br>분석한 비밀번호의 {pct:.0%}보다 내 비밀번호({d["L"]}자)가 깁니다.</div>')
        with c3:
            tips = []
            if (d["cnt"] and d["cnt"] > 0) or d["common"]: tips.append("이미 알려진 비밀번호입니다. 지금 쓰고 있다면 바로 바꾸세요.")
            if d["L"] < 12: tips.append("12자 이상, 가능하면 16자 이상으로 늘리세요. 길이가 가장 큰 영향을 줍니다.")
            if d["seq"] or d["rep"] or d["yr"] or d["allnum"]: tips.append("연속 문자, 반복 문자, 연도, 숫자만으로 된 구성은 피하세요.")
            if not (d["up"] and d["dg"] and d["sp"]): tips.append("기호를 억지로 섞기보다 서로 관련 없는 단어 4개 이상을 이은 문장형이 기억하기 쉽고 강합니다.")
            tips += ["사이트마다 다른 비밀번호를 쓰고 비밀번호 관리자를 사용하세요.", "가능한 모든 계정에 2단계 인증을 켜세요."]
            md('<div class="box"><b>이렇게 바꿔 보세요</b><br>' + "<br>".join(f"• {t}" for t in tips) + "</div>")
    with st.expander("계산 기준 · 한계"):
        st.markdown("**유출 확인**: Have I Been Pwned의 Pwned Passwords range API(키 불필요)를 사용합니다. SHA-1 값의 앞 5글자만 보내고 나머지 대조는 앱 서버에서 하며, 비밀번호 원문은 저장하지 않습니다. 이메일 계정 유출 조회용 v3 API는 유료 키가 필요해 쓰지 않았습니다.\n\n**점수**: 길이×log2(문자 종류 수)로 추정한 값에 패턴 감점을 적용해 80bit를 100점으로 환산했고, 유출·흔한 비밀번호는 상한을 둡니다. 교육용 근사치이며 실제 안전성을 보증하지 않습니다.")

with st.container(key=f"pg{page}"):
    _bd[0] = 0; [p_intro, p_leak, p_common, p_check][page]()
