import html as _html, inspect, math, re
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

try:
    st._config.set_option("theme.base", "light")
except Exception:
    pass

st.set_page_config(page_title="비밀번호는 얼마나 버틸까?", page_icon="🔐", layout="wide", initial_sidebar_state="collapsed")

D = Path(__file__).parent / "data"
FONT = "Pretendard, -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
BLUE, GREEN, ORANGE, RED, GRAY, YEL = "#5b8def", "#4cc38a", "#ff9f0a", "#ff3b30", "#86868b", "#ffcc00"
KO = {"Very week": "매우 약함", "Week": "약함", "Average": "보통", "Strong": "강함", "Very strong": "매우 강함"}
ORDER = list(KO.values())
CCOL = dict(zip(ORDER, [RED, ORANGE, YEL, BLUE, GREEN]))

CAT_KO = {
    "name": "이름·고유명사",
    "cool-macho": "허세·마초형",
    "simple-alphanumeric": "단순 숫자·키보드",
    "fluffy": "감성·애칭형",
    "sport": "스포츠·팀명",
    "nerdy-pop": "덕후·팝컬처",
    "animal": "동물",
    "password-related": "비밀번호·로그인류",
    "food": "음식",
    "rebellious-rude": "반항·비속어"
}

YR = 31557600
CAND = [1e-3, 1, 60, 3600, 86400, YR, YR * 1e2, YR * 1e4, YR * 1e6, YR * 1e9, YR * 1e12, YR * 1e18, YR * 1e24, YR * 1e36]

SRC = {
    "t1": (
        "Password Metrics Dataset (Kaggle)",
        "원본 100,000행 중 중복 비밀번호를 제거한 고유 25,000개(개인정보 보호를 위해 비밀번호 원문은 제거하고 길이·문자 구성·패턴 등 20개 수치 피처만 저장). 강도 등급은 등급별 약 5,000개씩 균등 추출된 표본이므로 실제 이용자 비율과 다름. 뚫리는 시간은 10^-7~10^104초로 범위가 넓어 로그 눈금·중앙값으로 집계.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset")]
    ),
    "t2": (
        "Password Strength and Vulnerability Dataset (Kaggle / Information is Beautiful)",
        "흔한 비밀번호 500개(인기 순위 기준). 원출처는 Information is Beautiful의 공개 자료입니다. value·time_unit은 '온라인 로그인 추측 공격으로 뚫는 시간', offline_crack_sec는 '유출 해시 파일 오프라인 대입 공격 시간(초)', strength는 '문자 구성 기반 기계적 강도 점수(0~48점)'이며 동일 초 단위로 환산해 비교했습니다.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords")]
    ),
    "t3": (
        "자체 학습 예측 모델 & 공개 데이터 검증 기준",
        "외부 API를 사용하지 않고, Kaggle Password Metrics Dataset(고유 25,000건)의 20개 구조적 피처로 학습한 Random Forest 회귀 모델과 상위 500개 흔한 비밀번호 목록(Information is Beautiful)을 결합해 예상 버팀 시간과 보안 등급을 산출합니다.",
        [
            ("학습 데이터 출처: Password Metrics Dataset (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset"),
            ("흔한 목록 출처: Password Strength Dataset (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords")
        ]
    )
}

FEAT_COLS = [
    "length", "n_lower", "n_upper", "n_digit", "n_special", "n_non",
    "r_lower", "r_upper", "r_digit", "r_special", "n_classes",
    "unique_ratio", "shannon", "longest_seq", "longest_rep",
    "has_year", "has_date", "has_leet", "digit_only", "alpha_only"
]

SEQ_ROWS = ["abcdefghijklmnopqrstuvwxyz", "0123456789", "qwertyuiop", "asdfghjkl", "zxcvbnm"]

def extract_features(pw: str) -> dict:
    L = len(pw)
    low = pw.lower()
    n_lower = sum(1 for c in pw if "a" <= c <= "z")
    n_upper = sum(1 for c in pw if "A" <= c <= "Z")
    n_digit = sum(1 for c in pw if "0" <= c <= "9")
    n_non = sum(1 for c in pw if ord(c) > 127)
    n_special = max(0, L - n_lower - n_upper - n_digit - n_non)
    denom = max(L, 1)

    n_classes = int(n_lower > 0) + int(n_upper > 0) + int(n_digit > 0) + int((n_special + n_non) > 0)
    unique_ratio = len(set(pw)) / denom

    if L > 0:
        counts = np.array([pw.count(c) for c in set(pw)], dtype=float)
        probs = counts / L
        shannon = float(-np.sum(probs * np.log2(probs)))
    else:
        shannon = 0.0

    longest_seq = 1
    for i in range(L):
        for j in range(i + 2, L + 1):
            sub = low[i:j]
            if any(sub in r or sub[::-1] in r for r in SEQ_ROWS):
                longest_seq = max(longest_seq, j - i)

    longest_rep = 1
    cur_rep = 1
    for i in range(1, L):
        if pw[i] == pw[i - 1]:
            cur_rep += 1
            longest_rep = max(longest_rep, cur_rep)
        else:
            cur_rep = 1

    has_year = int(bool(re.search(r"(19|20)\d\d", pw)))
    has_date = int(bool(re.search(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", pw)))
    has_leet = int(bool(re.search(r"[a-zA-Z]", pw) and re.search(r"[@$!013457]", pw)))
    digit_only = int(L > 0 and n_digit == L)
    alpha_only = int(L > 0 and (n_lower + n_upper) == L)

    return {
        "length": L, "n_lower": n_lower, "n_upper": n_upper, "n_digit": n_digit,
        "n_special": n_special, "n_non": n_non,
        "r_lower": n_lower / denom, "r_upper": n_upper / denom,
        "r_digit": n_digit / denom, "r_special": n_special / denom,
        "n_classes": n_classes, "unique_ratio": unique_ratio, "shannon": shannon,
        "longest_seq": longest_seq, "longest_rep": longest_rep,
        "has_year": has_year, "has_date": has_date, "has_leet": has_leet,
        "digit_only": digit_only, "alpha_only": alpha_only
    }

def md(s, pad=True):
    st.markdown("".join(l.strip() for l in s.splitlines()) + ('<div style="height:16px"></div>' if pad else ""), unsafe_allow_html=True)

def human(s):
    if s is None or not np.isfinite(s): return "측정 불가"
    if s < 1e-3: return "즉시 (1ms 미만)"
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
    kw = dict(tickvals=keep, ticktext=[human(10 ** x).replace(" (1ms 미만)", "") for x in keep], range=[lo - .4, hi + .4], tickangle=0)
    if axis == "y": fig.update_yaxes(**kw)
    else: fig.update_xaxes(**kw)

def style(fig, h=310, t=26, l=12, r=16):
    fig.update_layout(
        height=h, margin=dict(l=l, r=r, t=t, b=14), paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family=FONT, size=12, color="#1d1d1f"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        hoverlabel=dict(font_family=FONT, bgcolor="white")
    )
    fig.update_xaxes(gridcolor="#ececf0", zeroline=False, showline=False, automargin=True)
    fig.update_yaxes(gridcolor="#ececf0", zeroline=False, showline=False, automargin=True)
    return fig

_PW = "width" in inspect.signature(st.plotly_chart).parameters
def show(fig, key):
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key, **(dict(width="stretch") if _PW else dict(use_container_width=True)))

_bd = [0]
def head(title, take, table_fn=None):
    b = ""
    if title.startswith("질문"):
        _bd[0] = 0
    else:
        _bd[0] += 1
        b = f'<span class="bd">분석 {_bd[0]}</span>'

    if table_fn is None:
        md(f'<div class="sec"><div class="t">{b}{title}</div><div class="k">{take}</div></div>')
    else:
        c1, c2 = st.columns([7.8, 2.2], vertical_alignment="top")
        with c1:
            md(f'<div class="sec" style="min-height:auto"><div class="t">{b}{title}</div></div>', pad=False)
        with c2:
            with st.popover("📋 표", use_container_width=True):
                table_fn()
        md(f'<div class="sec" style="min-height:46px;padding-top:0"><div class="k" style="margin-top:0">{take}</div></div>')

def source(key):
    n, t, links = SRC[key]
    with st.expander("데이터 출처 · 집계 기준 · 한계"):
        link_md = " &nbsp;|&nbsp; ".join(f"🔗 [{label}]({url})" for label, url in links)
        st.markdown(f"**{n}**\n\n{t}\n\n{link_md}")

def ins(text, color=BLUE, label="핵심 결론"):
    md(f'<div class="ins" style="border-left-color:{color}"><b>{label}</b> {text}</div>')

def tile_grid(qq):
    lo, hi = 0.0, 7.5
    A, B, Cc = (255, 154, 143), (255, 228, 154), (143, 220, 176)
    def col(v):
        vc = float(np.clip(v, lo, hi))
        t = (vc - lo) / (hi - lo)
        p, q_, u = (A, B, t * 2) if t < .5 else (B, Cc, (t - .5) * 2)
        return "rgb(%d,%d,%d)" % tuple(int(p[k] + (q_[k] - p[k]) * u) for k in range(3))
    cells = "".join(
        f'<div class="tl" style="background:{col(v)};animation-delay:{min(i,20)*30}ms" title="{_html.escape(str(w))} ({_html.escape(str(ck))}) · 온라인 {human(s_)} / 오프라인 {human(of_)}">'
        f'<span class="rk">{int(r)}위</span><b>{_html.escape(str(w))}</b><span class="tm">{human(s_)}</span></div>'
        for i, (r, w, v, s_, of_, ck) in enumerate(zip(qq["rank"], qq.password, qq.lon, qq.online_sec, qq.offline_crack_sec, qq.cat_ko))
    )
    md(f'<div class="tg">{cells}</div>')

def sup(n):
    return str(int(n)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))

def infographic(cards):
    cs = "".join(
        f'<div class="ig" style="animation-delay:{i*90}ms"><div class="ic">{ic}</div><div class="it"><div class="ib">{big}<small>{u}</small></div><div class="icp">{cap}</div></div></div>'
        for i, (ic, big, u, cap) in enumerate(cards)
    )
    md(f'<div class="igs">{cs}</div>')

def kpis(items):
    cards = "".join(
        f'<div class="c" style="animation-delay:{i*70}ms"><div class="l">{l}</div>'
        f'<div class="v" style="font-size:{26 if isinstance(v,(int,float)) else 18}px">'
        f'<span class="n" data-v="{v if isinstance(v,(int,float)) else ""}" data-d="{d}">'
        f'{v if not isinstance(v,(int,float)) else f"{v:,.{d}f}"}</span><span class="u">{u}</span></div></div>'
        for i, (l, v, d, u) in enumerate(items)
    )
    h = f"""<style>
    body{{margin:0;padding:4px 2px 10px;font-family:{FONT}}}.g{{display:grid;grid-template-columns:repeat({len(items)},minmax(0,1fr));gap:12px}}
    .c{{background:#fff;border-radius:18px;padding:14px 16px 16px;border-left:5px solid #4cc38a;box-shadow:0 2px 12px rgba(0,0,0,.06);animation:u .5s ease both;min-width:0}}
    .l{{color:#6e6e73;font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .v{{margin-top:6px;color:#1d1d1f;font-weight:700;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.25}}.u{{font-size:14px;color:#6e6e73;margin-left:3px}}
    @keyframes u{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}</style>
    <div class="g">{cards}</div><script>document.querySelectorAll('.n').forEach(e=>{{const t=parseFloat(e.dataset.v);if(isNaN(t))return;const d=+e.dataset.d,s=performance.now();
    const f=n=>{{const p=Math.min((n-s)/900,1),x=1-Math.pow(1-p,3);e.textContent=(t*x).toLocaleString('en-US',{{minimumFractionDigits:d,maximumFractionDigits:d}});if(p<1)requestAnimationFrame(f)}};requestAnimationFrame(f)}})</script>"""
    try: components.html(h, height=114)
    except Exception: md("".join(f"<b>{l}</b> {v}{u} &nbsp; " for l, v, d, u in items))

def _find_file(name):
    for base in [D, Path(__file__).parent]:
        if (base / name).exists(): return base / name
    return D / name

@st.cache_data
def load():
    m = pd.read_csv(_find_file("features_unique.csv"))
    m["등급"] = pd.Categorical(m.class_strength.map(KO), ORDER, ordered=True)
    m["lc"] = np.log10(m.crack_time_sec.clip(lower=1e-12))
    conds = [
        m["digit_only"] == 1,
        m["alpha_only"] == 1,
        m["n_special"] > 0
    ]
    choices = ["숫자만", "영문만", "특수문자 포함"]
    m["조합유형"] = np.select(conds, choices, default="영문+숫자")

    p = pd.read_csv(_find_file("common_passwords.csv"))
    p["password"] = p["password"].astype(str)
    p["lon"] = np.log10(p.online_sec.clip(lower=1e-6))
    p["lof"] = np.log10(p.offline_crack_sec.clip(lower=1e-6))
    p["cat_ko"] = p["category"].map(CAT_KO).fillna(p["category"])
    return m, p

@st.cache_resource
def train_model(m: pd.DataFrame):
    shuffled = m.sample(frac=1.0, random_state=42)
    X = shuffled[FEAT_COLS]
    y = shuffled["lc"]
    model = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

M, P = load()
ML_MODEL = train_model(M)
TOP = set(P.password.str.lower())

md("""<style>@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html,body,[class*="css"],.stApp{font-family:Pretendard,-apple-system,'Apple SD Gothic Neo',sans-serif!important;overflow-x:hidden}
.stApp{background:#e9edf2}#MainMenu,footer{visibility:hidden}[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
.block-container{max-width:1520px;width:100%;box-sizing:border-box;padding:1rem 1.4rem 1.4rem;position:relative;z-index:1;animation:up .6s both}[data-testid="stVerticalBlock"]{gap:.55rem}
@keyframes up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes wipe{from{opacity:0;clip-path:inset(0 100% 0 0 round 22px)}to{opacity:1;clip-path:inset(0 0 0 0 round 22px)}}

/* 상단 헤더 1픽셀 수직 정중앙 정렬 */
.st-key-hdr{background:linear-gradient(100deg,#171f3d,#0f1730);border-radius:22px;padding:12px 22px;border:1px solid #25305a;box-shadow:0 6px 24px rgba(15,23,48,.25)}
.st-key-hdr [data-testid="stHorizontalBlock"]{align-items:center!important;min-height:46px}
.st-key-hdr [data-testid="stColumn"]{display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stElementContainer"],.st-key-hdr [data-testid="stMarkdownContainer"],.st-key-hdr [data-testid="stButton"],.st-key-hdr [data-testid="stButtonGroup"]{margin:0!important;padding:0!important;display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stMarkdownContainer"] p{margin:0!important;padding:0!important;line-height:46px!important}
.st-key-hdr *{color:#e6e9f2}
.ht{font-size:19.5px;font-weight:800;letter-spacing:-.02em;white-space:nowrap;display:inline-flex;align-items:center;gap:8px;height:46px;line-height:46px}
.st-key-hdr [data-testid="stButtonGroup"]{flex-wrap:nowrap;gap:8px;height:46px}
.st-key-hdr [data-testid="stButtonGroup"] button{height:46px!important;min-height:46px!important;padding:0 20px!important;font-size:15px;font-weight:700;border-radius:14px;background:#1a2547;border:1px solid #2a3868;white-space:nowrap;transition:all .25s;display:inline-flex;align-items:center;justify-content:center}
.st-key-hdr [data-testid="stButtonGroup"] button:hover{transform:translateY(-1px);border-color:#5b8def}
.st-key-hdr [data-testid="stButtonGroup"] button[aria-checked="true"],.st-key-hdr [data-testid="stButtonGroup"] button[aria-pressed="true"],.st-key-hdr button[data-testid="stBaseButton-segmented_controlActive"]{background:#5b8def!important;border-color:#8fb4ff;box-shadow:0 0 18px rgba(91,141,239,.55)}
.st-key-hdr [role="radiogroup"]{flex-direction:row;gap:8px;flex-wrap:nowrap;align-items:center}.st-key-hdr [role="radiogroup"] label{padding:8px 16px;border-radius:14px;background:#1a2547;border:1px solid #2a3868}
.st-key-hdr button[kind="secondary"]{background:#1a2547;border:1px solid #2a3868;border-radius:14px;height:46px!important;min-height:46px!important;padding:0 16px!important}

.sec{padding:4px 2px 4px;min-height:72px}.sec .t{font-size:16px;font-weight:800;letter-spacing:-.02em;line-height:1.35}.sec .k{font-size:12.5px;color:#5b6472;margin-top:4px;line-height:1.5}
.bd{background:#5b8def;color:#fff;font-size:11px;font-weight:700;border-radius:6px;padding:2px 8px;margin-right:8px;vertical-align:2px}
div[data-testid="stPlotlyChart"]{background:#fff;border-radius:22px;overflow:hidden;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:wipe .9s cubic-bezier(.2,.8,.2,1) both;transition:box-shadow .3s}
div[data-testid="stPlotlyChart"]:hover{box-shadow:0 14px 34px rgba(60,90,140,.16)}
.ins{background:#fff;border-left:5px solid #5b8def;border-radius:14px;padding:12px 18px;font-size:14px;line-height:1.6;box-shadow:0 2px 12px rgba(0,0,0,.05);animation:up .6s both}.ins b{margin-right:8px}

/* 질문 3 결과 카드 */
.res-card{background:#fff;border-radius:22px;padding:20px 22px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .6s both;min-height:462px;display:flex;flex-direction:column;justify-content:space-between;box-sizing:border-box}
.res-hdr{font-size:16.5px;font-weight:800;color:#1d1d1f;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between}
.stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:16px}
.stat-box{background:#f5f7fa;border-radius:16px;padding:14px 16px;border:1px solid #e8ecf2}
.stat-lbl{font-size:12px;font-weight:600;color:#6e6e73;margin-bottom:4px}
.stat-val{font-size:21px;font-weight:800;color:#1d1d1f;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-sub{font-size:11.5px;color:#5b6472;margin-top:4px}
.chk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:8px}
.chip{display:flex;align-items:center;justify-content:center;border-radius:10px;padding:9px 6px;font-size:12.5px;font-weight:700;text-align:center;white-space:nowrap}.ok{background:#e6f7ed;color:#157332;border:1px solid #c6ebd4}.no{background:#ffeceb;color:#c92a20;border:1px solid #ffd1ce}
.tip-item{background:#f8f9fc;border-radius:12px;padding:12px 14px;margin-bottom:10px;font-size:13.2px;line-height:1.55;color:#1d1d1f;border-left:4px solid #5b8def;word-break:keep-all}
.tip-item.warn{border-left-color:#ff3b30;background:#fff8f7}
.tip-item.good{border-left-color:#4cc38a;background:#f4fbf7}
.tip-tag{display:inline-block;font-size:11px;font-weight:800;padding:1px 7px;border-radius:6px;margin-right:6px;background:#e5edff;color:#2b5fc7}
.tip-item.warn .tip-tag{background:#ffe3e0;color:#c92a20}
.tip-item.good .tip-tag{background:#dcf5e6;color:#157332}

span[data-baseweb="tag"]{background:#5b8def!important;border-radius:8px}[role="slider"]{background:#5b8def!important}[data-testid="stExpander"]{border-radius:14px;background:#fff}
[data-testid="stPopover"] button{border-radius:99px!important;padding:2px 10px!important;min-height:30px!important;font-size:12px!important;font-weight:700!important;background:#fff!important;border:1px solid #d5dbe5!important;box-shadow:0 1px 4px rgba(0,0,0,.04)!important}
[data-testid="stPopover"] button:hover{border-color:#5b8def!important;color:#5b8def!important}
.spl{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.6s ease forwards}
.spl div{color:#fff;font-size:clamp(30px,6vw,64px);font-weight:800;letter-spacing:-.03em;animation:si 1.4s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.92);filter:blur(10px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,72%{opacity:1}100%{opacity:0;visibility:hidden}}
.tg{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;background:#fff;border-radius:22px;padding:14px;box-shadow:0 2px 14px rgba(0,0,0,.05);height:310px;overflow-y:auto;box-sizing:border-box;align-content:start}.tg::-webkit-scrollbar{width:6px}.tg::-webkit-scrollbar-thumb{background:#cfd6e4;border-radius:6px}.tl{border-radius:14px;padding:10px 6px;text-align:center;min-width:0;display:flex;flex-direction:column;gap:3px;justify-content:center;animation:up .4s both;transition:transform .25s}.tl:hover{transform:translateY(-3px) scale(1.03)}.tl b{font-size:14px;color:#1d1d1f;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rk{font-size:11px;color:rgba(0,0,0,.55);font-weight:600}.tm{font-size:12px;font-weight:700;color:rgba(0,0,0,.75);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.igs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.ig{display:flex;gap:14px;align-items:center;border-radius:22px;padding:16px 18px;color:#fff;animation:up .6s both;min-width:0;box-shadow:0 8px 22px rgba(60,90,140,.18);transition:transform .25s}.ig:hover{transform:translateY(-4px)}.ig:nth-child(1){background:linear-gradient(135deg,#4f7fe6,#7aa7ff)}.ig:nth-child(2){background:linear-gradient(135deg,#f0554a,#ff8f7a)}.ig:nth-child(3){background:linear-gradient(135deg,#e8860a,#ffb347)}.ig:nth-child(4){background:linear-gradient(135deg,#2ea36f,#6fd6a3)}.ic{font-size:36px;flex:none;filter:drop-shadow(0 4px 6px rgba(0,0,0,.18))}.ib{font-size:28px;font-weight:800;letter-spacing:-.02em;line-height:1.1;white-space:nowrap}.ib small{font-size:13px;font-weight:700;margin-left:3px;opacity:.92}.icp{font-size:12px;line-height:1.45;margin-top:5px;opacity:.96}
html{color-scheme:light}[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] strong,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *,[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"],[data-testid="stCheckbox"] label *,[data-testid="stToggle"] label *{color:#1d1d1f!important}.sec .t,.ins,.ins b{color:#1d1d1f}.sec .k{color:#5b6472}.ins{color:#1d1d1f}div.st-key-hdr,div.st-key-hdr *,div.st-key-hdr p,div.st-key-hdr button *{color:#e6e9f2!important}div.ig,div.ig *{color:#fff!important}
.st-key-chk_panel{background:#fff;border-radius:22px;padding:14px 18px;box-shadow:0 2px 14px rgba(0,0,0,.05)}
.st-key-chk_panel [data-testid="stHorizontalBlock"]{align-items:center!important}
.st-key-d_ex0 button,.st-key-d_ex1 button,.st-key-d_ex2 button{width:100%;min-height:44px;padding:0 10px!important;border-radius:14px;border:0;box-shadow:0 4px 12px rgba(60,90,140,.16);transition:transform .2s,box-shadow .2s;white-space:nowrap}.st-key-d_ex0 button{background:linear-gradient(135deg,#f0554a,#ff8f7a)}.st-key-d_ex1 button{background:linear-gradient(135deg,#e8860a,#ffb347)}.st-key-d_ex2 button{background:linear-gradient(135deg,#2ea36f,#6fd6a3)}.st-key-d_ex0 button:hover,.st-key-d_ex1 button:hover,.st-key-d_ex2 button:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(60,90,140,.24)}.st-key-d_ex0 button *,.st-key-d_ex1 button *,.st-key-d_ex2 button *{color:#fff!important;font-size:13px;font-weight:700}
[data-testid="stForm"] button{background:#5b8def;border:0;border-radius:14px;min-height:44px;width:100%}[data-testid="stForm"] button *{color:#fff!important;font-weight:700;font-size:13.5px}[data-testid="stForm"] input{border-radius:14px;min-height:44px}[data-testid="InputInstructions"]{display:none!important}</style>""", False)

if not st.session_state.get("splash"):
    st.session_state["splash"] = True
    md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>', False)

NAMES = ["👋 시작하기", "📊 뚫리는 시간", "⚠️ 흔한 비밀번호", "🔍 내 비밀번호 점검"]
if "nav" not in st.session_state: st.session_state["nav"] = NAMES[0]
def _keep():
    if st.session_state.get("nav") is None: st.session_state["nav"] = st.session_state.get("last_nav", NAMES[0])

with st.container(key="hdr"):
    hc = st.columns([2.4, 8.2, 1.2], vertical_alignment="center")
    hc[0].markdown('<div class="ht">🔐 비밀번호 보안 분석</div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        hc[1].segmented_control("메뉴", NAMES, key="nav", label_visibility="collapsed", on_change=_keep)
    else:
        hc[1].radio("메뉴", NAMES, horizontal=True, label_visibility="collapsed", key="nav")
    hc[2].button("▶ 인트로", on_click=lambda: st.session_state.update(splash=False), use_container_width=True)

cur = st.session_state.get("nav") or NAMES[0]
st.session_state["last_nav"] = cur
page = NAMES.index(cur)

def _setall(key, vals): st.session_state[key] = list(vals)
def multi_pick(col, label, options, key):
    if key not in st.session_state: st.session_state[key] = list(options)
    with col.popover(f"{label} · {len(st.session_state[key])}/{len(options)} 선택"):
        b = st.columns(2)
        b[0].button("모두 선택", key=f"{key}_all", on_click=_setall, args=(key, options))
        b[1].button("모두 해제", key=f"{key}_none", on_click=_setall, args=(key, []))
        if hasattr(st, "pills"): return st.pills(label, options, selection_mode="multi", key=key, label_visibility="collapsed") or []
        return st.multiselect(label, options, key=key, label_visibility="collapsed")

def filters(n=3): return st.columns([2, 2, 3][:n], vertical_alignment="center")

def p_intro():
    head("질문 0. 비밀번호가 '뚫린다'는 건 무슨 뜻일까?", "처음 보시는 분을 위한 요약판입니다. 아래 기준을 바꾸면 숫자와 차트가 함께 바뀝니다.")
    fl = filters(); TS = {"1분": 60, "1시간": 3600, "1일": 86400, "1년": YR, "100년": YR * 100}
    tk = fl[0].select_slider("이 시간 안에 뚫리면 '위험'으로 셉니다", options=list(TS), value="1년", key="i_t"); T = TS[tk]
    Ln = fl[1].slider("내 비밀번호 길이(자)", 4, 24, 12, key="i_l")
    fl[2].caption("↻ 기준을 바꾸면 아래 숫자와 차트가 바뀝니다")

    pc = [(M.crack_time_sec <= T).mean() * 100, (P.online_sec <= T).mean() * 100, (P.offline_crack_sec <= T).mean() * 100]
    ins(
        f"해킹 프로그램은 비밀번호를 하나씩 대입하며 맞는 값을 찾습니다. 같은 기준({tk})으로 보면 "
        f"균등 추출 표본 기준 <b>{pc[0]:.0f}%</b>(실제 이용자 분포와 다름), 흔한 비밀번호 목록은 온라인 <b>{pc[1]:.0f}%</b>·오프라인 <b>{pc[2]:.0f}%</b>가 해당 시간 안에 뚫립니다. "
        f"공격 경로(온라인 로그인 창 vs 유출 파일 오프라인 해독)에 따라 위험도가 극명하게 갈립니다.",
        GREEN, "한눈에 보기"
    )

    gap = float((P.lon - P.lof).median())
    slp = float(np.polyfit(M.length, M.lc, 1)[0])
    infographic([
        ("📦", f"{len(M) + len(P):,}", "개", f"분석 데이터 (고유 피처 표본 {len(M):,}개 + 흔한 목록 {len(P)}개)"),
        ("🔓", f"{pc[1]:.0f}", "%", f"흔한 비밀번호 중 {tk} 안에 로그인 창 공격으로 뚫리는 비율"),
        ("⚡", f"10{sup(round(gap))}", "배", "오프라인 유출 파일 대입이 온라인 로그인 공격보다 빠른 배율(중앙값)"),
        ("📏", f"×{10 ** slp:.0f}", "배 / 1자", "비밀번호가 1자 길어질 때 뚫리는 시간이 늘어나는 평균 배율")
    ])

    x1, x2, x3 = st.columns([3.5, 3.2, 3.3])
    with x1:
        head("100개 중 몇 개가 뚫릴까?(와플 차트)", f"네모 1칸 = 1%, 빨강은 {tk} 안에 뚫리는 비율입니다. 유출 표본은 등급별 균등 표본이라 실제 분포가 아닙니다.")
        ttl = ["균등 추출 표본", "흔한·온라인", "흔한·오프라인"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t}<br><b>{p:.0f}%</b>" for t, p in zip(ttl, pc)])
        for i, p in enumerate(pc):
            z = np.zeros(100); z[:int(round(p))] = 1
            fig.add_trace(go.Heatmap(z=z.reshape(10, 10), colorscale=[[0, "#cfeedd"], [1, "#ff6b6b"]], showscale=False, xgap=3, ygap=3, hoverinfo="skip"), 1, i + 1)
            fig.update_xaxes(visible=False, row=1, col=i + 1); fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i + 1)
        fig.update_annotations(font_size=11)
        show(style(fig, 310, t=64).update_layout(margin=dict(l=12, r=12, t=64, b=12)), "i1")

    with x2:
        L_ref = max(Ln, 10)
        sub = M[M.length.between(L_ref - 1, L_ref + 1)]
        if len(sub) < 60:
            sub = M[M.length.between(max(4, L_ref - 2), L_ref + 2)]
        clean = sub[(sub.longest_seq < 3) & (sub.longest_rep < 3) & (sub.has_year == 0) & (sub.has_date == 0)]
        note = " (8자 이하는 모두 즉시 뚫려 최소 10자 기준으로 비교)" if Ln < 10 else ""
        head(
            f"같은 {L_ref}자 내외, 사소한 습관이 깎는 시간(막대)",
            f"길이를 늘려도 연속키(qwer·123), 생일·연도, 뻔한 치환(@·0)이 섞이면 방어력이 급감합니다.{note}"
        )
        leet_sub = sub[(sub.has_leet == 1) & ((sub.longest_seq >= 3) | (sub.has_year == 1) | (sub.n_classes <= 3))]
        groups = [
            ("연속·반복 (qwer/aaa)", sub[(sub.longest_seq >= 3) | (sub.longest_rep >= 3)].crack_time_sec.median(), RED),
            ("연도·날짜 (2024/0315)", sub[(sub.has_year == 1) | (sub.has_date == 1)].crack_time_sec.median(), ORANGE),
            ("알파벳 치환 (P@ssw0rd류)", (leet_sub if not leet_sub.empty else sub[sub.has_leet == 1]).crack_time_sec.median(), BLUE),
            ("패턴 없는 무작위 조합", (clean[clean.n_classes >= 3] if not clean[clean.n_classes >= 3].empty else clean).crack_time_sec.median(), GREEN),
        ]
        y_lbls = [g[0] for g in groups]
        meds = [g[1] for g in groups]
        cols = [g[2] for g in groups]
        l_meds = [float(np.clip(math.log10(max(v, 1e-3)), -3, 16)) if pd.notna(v) else -3.0 for v in meds]
        txt_lbl = [human(v).replace(" (1ms 미만)", "") if pd.notna(v) else "표본 없음" for v in meds]

        fig = go.Figure(go.Bar(
            y=y_lbls, x=[v + 3.2 for v in l_meds], orientation="h",
            marker_color=cols, text=txt_lbl, textposition="outside", cliponaxis=False,
            textfont=dict(size=12, weight="bold", color="#1d1d1f"),
            hovertemplate="%{y}: %{text}<extra></extra>"
        ))
        fig.update_xaxes(
            tickvals=[0.2, 3.2, 6.756, 8.136, 10.699, 12.699],
            ticktext=["1ms", "1초", "1시간", "1일", "1년", "100년"],
            range=[0, max(l_meds) + 6.8]
        )
        fig.update_yaxes(showgrid=False)
        show(style(fig, 310, t=24, l=12, r=28), "i2")

    with x3:
        head("길이별 버팀 시간 추세(회귀 추세선)", "파란 점은 길이별 중앙값, 점선은 지수 회귀 추세선입니다. 빨간 점선(1년) 위에 머무르려면 최소 11~12자 이상이 필요합니다.")
        g_i = M[M.length <= 24].groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=15").reset_index()
        y_i = np.log10(g_i.med.clip(lower=1e-3))
        coef = np.polyfit(g_i.length, y_i, 1)
        r2_i = float(np.corrcoef(g_i.length, y_i)[0, 1] ** 2)
        y_fit = np.polyval(coef, g_i.length)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=g_i.length, y=y_i, mode="markers", name="길이별 중앙값",
            marker=dict(color=BLUE, size=8),
            customdata=[human(v).replace(" (1ms 미만)", "") for v in g_i.med],
            hovertemplate="%{x}자 중앙값: %{customdata}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=g_i.length, y=y_fit, mode="lines", name=f"회귀 추세선 (R²={r2_i:.2f})",
            line=dict(color="#1d1d1f", width=2.5, dash="dash"), hoverinfo="skip"
        ))
        fig.add_vline(x=Ln, line_color=GREEN, line_width=2, annotation_text=f"선택 {Ln}자", annotation_position="top right")
        fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=RED, annotation_text="안전 기준(1년)", annotation_position="bottom right")
        fig.update_xaxes(title="비밀번호 길이(자)")
        tick_axis(fig, min(y_i.min(), -2), max(y_i.max(), math.log10(YR) + 1))
        show(style(fig, 310, t=34), "i3")

    with st.expander("데이터 출처 요약"):
        st.markdown(
            "1. **Password Metrics Dataset** (고유 피처 25,000건 추출): [Kaggle 바로가기](https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset)\n"
            "2. **Password Strength and Vulnerability Dataset** (흔한 비밀번호 500건): [Kaggle 바로가기](https://www.kaggle.com/datasets/utkarshx27/passwords)"
        )

def p_leak():
    head("질문 1. 비밀번호가 길수록 얼마나 오래 버틸까?", "고유 비밀번호 피처 25,000개를 등급과 길이별로 살펴봅니다. (원본 데이터는 등급별 균등 추출 표본입니다)")
    fl = filters()
    gs = multi_pick(fl[0], "강도 등급", ORDER, "a_g")
    lr = fl[1].slider("길이 범위(자)", 4, 63, (4, 40), key="a_l")
    fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다")

    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty:
        st.info("조건에 맞는 데이터가 없습니다. 등급이나 길이 범위를 넓혀 보세요.")
        return

    ent_col = "entropy" if "entropy" in f.columns else "shannon"
    kpis([
        ("선택된 표본 수", len(f), 0, "개"),
        ("길이 중앙값", float(f.length.median()), 0, "자"),
        ("엔트로피 중앙값", float(f[ent_col].median()), 1, "bit"),
        ("뚫리는 시간 중앙값", human(f.crack_time_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("표본 내 1년 내 뚫림", (f.crack_time_sec <= YR).mean() * 100, 0, "%"),
        ("표본 내 1시간 내 뚫림", (f.crack_time_sec <= 3600).mean() * 100, 0, "%")
    ])

    g = f.groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=20").reset_index()
    if f.length.nunique() > 1:
        slope = np.polyfit(f.length, f.lc, 1)[0]
        r2 = float(np.corrcoef(f.length, f.lc)[0, 1] ** 2)
    else:
        slope, r2 = float("nan"), float("nan")

    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        desc = (
            f"길이가 1자 늘 때마다 뚫리는 시간이 평균 약 {10**slope:.1f}배 늘어납니다(선형 설명력 R²={r2:.2f}). 표본 20개 미만 길이는 제외했습니다."
            if np.isfinite(slope) else "길이 구간을 넓히면 추세가 보입니다."
        )
        tbl_a1 = (lambda: st.dataframe(g.assign(중앙값=g.med.map(human))[["length", "n", "중앙값"]].rename(columns={"length": "길이", "n": "표본"}), hide_index=True)) if not g.empty else None
        head("길이가 길수록 얼마나 오래 걸릴까?(선 그래프)", desc, table_fn=tbl_a1)
        if g.empty:
            st.info("표본이 부족합니다.")
        else:
            y = np.log10(g.med.clip(lower=1e-3))
            fig = go.Figure(go.Scatter(
                x=g.length, y=y, mode="lines+markers", line=dict(color=BLUE, width=3, shape="spline"), marker=dict(size=7),
                customdata=[[human(v), n] for v, n in zip(g.med, g.n)],
                hovertemplate="길이 %{x}자<br>중앙값 %{customdata[0]}<br>표본 %{customdata[1]}개<extra></extra>"
            ))
            fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=GRAY, annotation_text="1년", annotation_position="top left")
            fig.update_xaxes(title="비밀번호 길이(자)")
            tick_axis(fig, min(y.min(), math.log10(YR)), y.max())
            show(style(fig), "a1")

    with x2:
        stp = [("시작", 0), ("1분 버팀", 60), ("1시간 버팀", 3600), ("1일 버팀", 86400), ("1년 버팀", YR), ("100년 버팀", YR * 100)]
        vals = [100.0] + [(f.crack_time_sec > t).mean() * 100 for _, t in stp[1:]]
        head("시간이 지날수록 표본의 몇 %가 버틸까?(깔때기 차트)", f"선택한 균등 표본 중 {vals[3]:.0f}%가 1일을, {vals[4]:.0f}%가 1년을 버팁니다.")
        fig = go.Figure(go.Funnel(
            y=[a for a, _ in stp], x=vals, text=[f"{v:.0f}%" for v in vals], textinfo="text", textposition="inside",
            textfont=dict(size=13, color="#fff"),
            marker=dict(color=["#5b8def", "#6e9bf0", "#4cc38a", "#3bb27a", "#2a9d68", "#1f8a58"]),
            connector=dict(line=dict(color="#e5e7eb"))
        ))
        show(style(fig), "a2")

    with x3:
        head("문자 조합 유형별 뚫리는 시간 분포(박스 플롯)", "길이가 같아도 숫자만 쓴 경우보다 영문·숫자·특수문자를 섞을수록 방어 시간이 크게 상승합니다.")
        type_order = ["숫자만", "영문만", "영문+숫자", "특수문자 포함"]
        type_cols = [RED, ORANGE, BLUE, GREEN]
        fig = go.Figure()
        for tp, c_ in zip(type_order, type_cols):
            sub_t = f[f["조합유형"] == tp]
            if not sub_t.empty:
                fig.add_trace(go.Box(y=sub_t["lc"].clip(lower=-3), name=tp, marker_color=c_, boxpoints=False))
        tick_axis(fig, max(f["lc"].min(), -3), f["lc"].max(), "y")
        fig.update_layout(showlegend=False)
        show(style(fig), "a3")

    ins(f"선택한 균등 표본 {len(f):,}개 중 {(f.crack_time_sec <= YR).mean():.0%}가 1년 안에 뚫립니다. 길이 중앙값 {int(f.length.median())}자 표본의 뚫리는 시간 중앙값은 {human(f.crack_time_sec.median())}입니다.")
    source("t1")

def p_common():
    head("질문 2. 흔한 비밀번호는 왜 위험하고, 어떤 착시를 일으킬까?", "전 세계 상위 500개 흔한 비밀번호의 심리 유형과 '영문+숫자 조합'이 만드는 가짜 안전 점수(strength)의 함정을 살펴봅니다.")
    fl = filters()
    n = fl[0].slider("상위 몇 개를 볼까요?", 10, 500, 100, 10, key="b_n")
    allc = list(P.cat_ko.value_counts().index)
    cats = multi_pick(fl[1], "비밀번호 유형", allc, "b_c")
    fl[2].caption("↻ 필터를 바꾸면 KPI와 차트가 함께 바뀝니다 (500개로 넓히면 passw0rd·bond007 등도 확인 가능)")

    q = P[(P["rank"] <= n) & P.cat_ko.isin(cats)]
    if q.empty:
        st.info("조건에 맞는 비밀번호가 없습니다. 비밀번호 유형을 선택해 주세요.")
        return

    topc = q.cat_ko.value_counts()
    trap_q = q[q["strength"] >= 20]
    kpis([
        ("표시 중 비밀번호", len(q), 0, "개"),
        ("가장 많은 유형", f"{topc.index[0]}", 0, f" {topc.iloc[0]/len(q):.0%}"),
        ("기계 강도 20점+ 착시", len(trap_q), 0, "개"),
        ("온라인 공격 · 중앙값", human(q.online_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("오프라인 유출 · 중앙값", human(q.offline_crack_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("오프라인 · 1초 내 뚫림", (q.offline_crack_sec <= 1).mean() * 100, 0, "%")
    ])

    x1, x2, x3 = st.columns([4, 2.8, 3.2])
    with x1:
        tbl_b1 = lambda: st.dataframe(
            q.assign(온라인=q.online_sec.map(human), 오프라인=q.offline_crack_sec.map(human), 강도점수=q.strength.astype(int))[
                ["rank", "password", "cat_ko", "강도점수", "온라인", "오프라인"]
            ].rename(columns={"rank": "순위", "password": "비밀번호", "cat_ko": "유형"}),
            hide_index=True
        )
        head(f"가장 흔한 비밀번호 {len(q)}개 타일 지도", "온라인 시간 기준 색상(빨강=즉시~수분, 노랑=수일~수개월, 초록=1년+). 마우스를 올리면 오프라인 시간도 보입니다.", table_fn=tbl_b1)
        tile_grid(q)

    with x2:
        head("사람들은 어떤 단어를 제일 많이 쓸까?(도넛)", f"1위는 '{topc.index[0]}'({topc.iloc[0]}개, {topc.iloc[0]/len(q):.0%})입니다. 이름·허세·감성 단어가 대다수입니다.")
        fig = go.Figure(go.Pie(
            labels=topc.index, values=topc.values, hole=.58, sort=False, textinfo="percent",
            marker=dict(colors=px.colors.qualitative.Set3 * 2, line=dict(color="#fff", width=2)),
            hovertemplate="%{label}: %{value}개 (%{percent})<extra></extra>"
        ))
        fig.add_annotation(text=f"<b>{len(q)}개</b>", x=.5, y=.5, showarrow=False, font=dict(size=21))
        show(style(fig, 310).update_layout(legend=dict(orientation="v", x=1.0, y=.5, font=dict(size=10.5))), "b2")

    with x3:
        head(
            "기계는 안전하다는데 다 털리는 '착시 비밀번호'(산점도)",
            "영문+숫자를 섞어 강도 점수(Y축)는 높게 받았지만, 흔한 목록이라 사전 공격에 1초 만에 뚫리는 함정입니다."
        )
        norm_q = q[q["strength"] < 20]
        high_q = q[q["strength"] >= 20].sort_values("strength", ascending=False)

        fig = go.Figure()
        if not norm_q.empty:
            fig.add_trace(go.Scatter(
                x=norm_q["rank"], y=norm_q["strength"], mode="markers", name="일반 흔한 비번 (<20점)",
                marker=dict(color="#b8c2d1", size=7, opacity=0.65),
                customdata=np.stack([norm_q.password, norm_q.cat_ko, norm_q.offline_crack_sec.map(human)], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> (%{x}위 · %{customdata[1]})<br>기계 강도: %{y:.0f}점<br>오프라인 해독: %{customdata[2]}<extra></extra>"
            ))
        if not high_q.empty:
            top_lbl_set = set(high_q.head(6).password)
            txt_arr = [pw if pw in top_lbl_set else "" for pw in high_q.password]
            fig.add_trace(go.Scatter(
                x=high_q["rank"], y=high_q["strength"], mode="markers+text", name="고득점 착시 비번 (20점+)",
                text=txt_arr, textposition="top center", cliponaxis=False,
                textfont=dict(size=11, color=RED, weight="bold"),
                marker=dict(color=RED, size=10, line=dict(color="#fff", width=1.5)),
                customdata=np.stack([high_q.password, high_q.cat_ko, high_q.offline_crack_sec.map(human)], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b> (%{x}위 · %{customdata[1]})<br>기계 강도: %{y:.0f}점 (착시!)<br>오프라인 해독: %{customdata[2]}<extra></extra>"
            ))
        fig.add_hline(y=20, line_dash="dot", line_color=ORANGE, annotation_text="영문+숫자 착시선(20점)", annotation_position="bottom right")
        fig.update_xaxes(title="인기 순위 (1위 = 가장 흔함)")
        fig.update_yaxes(title="기계적 강도 점수 (strength)", range=[-3, max(float(q.strength.max()) + 10, 42)])
        show(style(fig, 310, t=34, r=20), "b3")

    ex_names = " · ".join(f"<b>{w}</b>({int(s)}점)" for w, s in zip(trap_q.sort_values("strength", ascending=False).password.head(4), trap_q.sort_values("strength", ascending=False).strength.head(4)))
    trap_msg = f" 특히 {ex_names}처럼 영문과 숫자를 섞어 기계 점수가 높더라도 유명 단어·팝컬처 조합이면 해커 사전 1순위에 올라 즉시 뚫립니다." if not trap_q.empty else ""
    ins(
        f"온라인 대입보다 유출 해시 오프라인 공격은 정확히 <b>1억 배(10⁸배)</b> 빠릅니다(오프라인 중앙값 {human(q.offline_crack_sec.median())}).{trap_msg}",
        RED, "착시 주의"
    )
    source("t2")

def diagnose(pw: str):
    feats = extract_features(pw)
    L = feats["length"]
    low = pw.lower()

    X_in = pd.DataFrame([feats])[FEAT_COLS]
    pred_lc = float(ML_MODEL.predict(X_in)[0])

    if feats["n_non"] > 0:
        pred_lc += min(feats["n_non"], 4) * 0.4
    seq = feats["longest_seq"] >= 3
    rep = feats["longest_rep"] >= 3
    yr = bool(feats["has_year"] or feats["has_date"])
    penalty_log = 1.5 * seq + 1.2 * rep + 1.0 * yr
    adj_lc = pred_lc - penalty_log
    est_sec = 10 ** adj_lc

    pct = float((M["lc"] <= adj_lc).mean() * 100)
    score = float(np.clip((adj_lc + 2.0) / 14.0 * 100.0, 0, 100))

    common = low in TOP
    if common:
        score = min(score, 5.0)
        est_sec = min(est_sec, 1.0)

    st.session_state["diag"] = dict(
        L=L, score=score, est_sec=est_sec, pct=pct, common=common,
        seq=seq, rep=rep, yr=yr, up=feats["n_upper"] > 0, dg=feats["n_digit"] > 0, sp=(feats["n_special"] + feats["n_non"]) > 0
    )

def p_check():
    head("질문 3. 내 비밀번호는 안전할까?", "입력한 비밀번호는 서버에 저장되지 않습니다. 고유 데이터 25,000건으로 학습한 ML 예측 모델과 흔한 비밀번호 500개 DB를 결합해 진단합니다.")

    with st.container(key="chk_panel"):
        md('<div style="font-size:13.5px;font-weight:700;color:#1d1d1f;margin-bottom:8px">직접 입력하거나, 오른쪽 예시 유형을 눌러 바로 테스트해 보세요</div>', pad=False)
        top_row = st.columns([4.6, 1.8, 1.8, 1.8], vertical_alignment="center")
        with top_row[0]:
            with st.form("f", clear_on_submit=True, border=False):
                fc = st.columns([3.8, 1.4], vertical_alignment="center")
                pw = fc[0].text_input("비밀번호 입력", type="password", placeholder="비밀번호를 입력하세요 (저장 안 됨)", label_visibility="collapsed")
                if fc[1].form_submit_button("점검하기") and pw:
                    diagnose(pw)
        for i, (col, (lab, v)) in enumerate(zip(top_row[1:], [("🔓 흔한형", "passw0rd"), ("🧩 규칙형", "Summer2024!"), ("🎲 무작위형", "vK9#mQ2$zL7@pW4xR8")])):
            if col.button(lab, key=f"d_ex{i}", use_container_width=True):
                diagnose(v)

    d = st.session_state.get("diag")
    if d:
        s = d["score"]
        grade, col = ("위험", RED) if s < 30 else ("주의", ORANGE) if s < 55 else ("양호", BLUE) if s < 80 else ("안전", GREEN)
        if d["common"]:
            grade, col = "위험", RED

        c1, c2, c3 = st.columns([2.3, 3.2, 2.5])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(s),
                number=dict(suffix="점", font=dict(size=38)),
                title=dict(text=f"진단 등급: {grade}", font=dict(size=16)),
                gauge=dict(
                    axis=dict(range=[0, 100]), bar=dict(color=col, thickness=.35), bgcolor="#f2f2f5", borderwidth=0,
                    steps=[dict(range=[0, 30], color="#ffe9e7"), dict(range=[30, 55], color="#fff2dc"), dict(range=[55, 80], color="#e5f1ff"), dict(range=[80, 100], color="#e3f8e9")]
                )
            ))
            show(style(fig, 235, t=54, l=22, r=24), "d1")
            vf = go.Figure([go.Violin(x=M[M.등급 == g_].length, name=g_, orientation="h", line_color=CCOL[g_], points=False, showlegend=False, spanmode="hard") for g_ in ORDER])
            vf.add_vline(x=d["L"], line_color="#1d1d1f", line_dash="dash", annotation_text="내 길이")
            vf.update_xaxes(title="표본 등급별 길이 분포(자)")
            show(style(vf, 215, t=22), "d2")

        with c2:
            chips = [
                ("8자 이상", d["L"] >= 8), ("12자 이상", d["L"] >= 12), ("대문자 포함", d["up"]),
                ("숫자 포함", d["dg"]), ("특수문자 포함", d["sp"]), ("흔한 목록 제외", not d["common"]),
                ("연속 문자 없음", not d["seq"]), ("반복 문자 없음", not d["rep"]), ("연도/날짜 없음", not d["yr"])
            ]
            ch = "".join(f'<div class="chip {"ok" if v else "no"}">{"✓" if v else "✕"} {k}</div>' for k, v in chips)

            md(f"""<div class="res-card">
                <div>
                    <div class="res-hdr">
                        <span>🤖 ML 예측 기반 정밀 진단 리포트</span>
                        <span style="font-size:12px;color:#5b6472;font-weight:600">Random Forest 예측</span>
                    </div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-lbl">예상 뚫리는 시간</div>
                            <div class="stat-val">{human(d['est_sec'])}</div>
                            <div class="stat-sub">구조·패턴 종합 산출</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">표본 내 보안 순위</div>
                            <div class="stat-val">상위 {max(0.1, 100 - d['pct']):.1f}%</div>
                            <div class="stat-sub">고유 25,000건 기준</div>
                        </div>
                    </div>
                </div>
                <div>
                    <div style="font-size:13.5px;font-weight:800;color:#1d1d1f;margin-bottom:6px">9대 보안 구성 체크리스트</div>
                    <div class="chk-grid">{ch}</div>
                </div>
            </div>""")

        with c3:
            tips = []
            if d["common"]:
                tips.append(("warn", "긴급", "전 세계 상위 500위 안에 드는 흔한 비밀번호입니다. 사전 대입 공격으로 1초 내에 뚫립니다."))
            if d["L"] < 12:
                tips.append(("norm", "길이 보강", f"현재 길이(<b>{d['L']}자</b>)는 다소 짧습니다. 최소 12자 이상으로 늘리면 방어 시간이 기하급수적으로 증가합니다."))
            if not (d["up"] and d["dg"] and d["sp"]):
                tips.append(("norm", "문자 조합", "대문자, 숫자, 특수문자를 고르게 섞어 한 자리당 탐색 경우의 수(Pool)를 넓히세요."))
            if d["seq"] or d["rep"] or d["yr"]:
                tips.append(("norm", "패턴 제거", "키보드 연속 배열(qwerty, 123), 반복 문자(aaa), 연도/생일(2024, 0315) 패턴을 제거하세요."))
            if not tips:
                tips.append(("good", "최상 등급", "길이와 문자 다양성이 모두 우수하며 취약 패턴도 없습니다. 중요 계정에는 2단계 인증(OTP/패스키)을 함께 켜두세요."))

            tip_html = "".join(f'<div class="tip-item {cls}"><span class="tip-tag">{tag}</span>{txt}</div>' for cls, tag, txt in tips)
            md(f"""<div class="res-card" style="justify-content:flex-start">
                <div class="res-hdr"><span>💡 맞춤 보안 처방 가이드</span></div>
                <div>{tip_html}</div>
            </div>""")

    source("t3")

if page == 0: p_intro()
elif page == 1: p_leak()
elif page == 2: p_common()
else: p_check()
