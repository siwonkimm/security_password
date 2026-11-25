import hashlib, html as _html, inspect, math, re, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px, streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

try:
    st._config.set_option("theme.base", "light")
except Exception:
    pass

st.set_page_config(page_title="비밀번호는 얼마나 버틸까?", page_icon="🔐", layout="wide", initial_sidebar_state="collapsed")

D = Path(__file__).parent / "data"
FONT = "Pretendard, -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"
BLUE, GREEN, ORANGE, RED, GRAY, YEL = "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#64748b", "#eab308"
KO = {"Very week": "매우 약함", "Week": "약함", "Average": "보통", "Strong": "강함", "Very strong": "매우 강함"}
ORDER = list(KO.values())
CCOL = dict(zip(ORDER, [RED, ORANGE, YEL, BLUE, GREEN]))
YR = 31557600
CAND = [1e-2, 1, 60, 3600, 86400, YR, YR * 100, YR * 1e4, YR * 1e8, YR * 1e12, YR * 1e18, YR * 1e24, YR * 1e36]

CAT_KO = {
    "name": "사람 이름",
    "cool-macho": "멋진·강한 단어",
    "simple-alphanumeric": "단순 영문+숫자",
    "fluffy": "일상·애칭 단어",
    "sport": "스포츠·운동",
    "nerdy-pop": "영화·게임·문화",
    "animal": "동물 이름",
    "password-related": "비밀번호 관련어",
    "food": "음식 이름",
    "rebellious-rude": "비속어·반항 단어"
}

SRC = {
    "t0": (
        "분석 데이터셋 통합 요약 (총 25,500건) · 계산 기준 및 한계",
        "① Password Metrics Dataset(고유 25,000건): 한 자리에 쓸 수 있는 글자 종류 수(N)와 길이(L)를 곱한 모든 경우의 수(N^L)를 무차별 대입할 때 걸리는 이론적 시간(초)입니다. "
        "[한계] 보안 등급별 약 5,000건씩 고르게 뽑은 균등 표본이므로 실제 사람들의 비밀번호 비율과는 다르며, 원본 수식에는 연도·키보드 배열 감점이 반영되어 있지 않습니다.\n\n"
        "② Password Strength Dataset(흔한 비밀번호 상위 500건): 로그인 창 직접 입력(초당 10회 가정)과 유출된 암호화 파일 고속 대입(초당 10억 회 가정, 10^8배=1억 배 빠름) 시간을 비교합니다. "
        "[한계] 실제 해커가 1순위로 쓰는 '사전 대입 공격(흔한 목록 먼저 넣기)' 앞에서는 500개 모두 표기된 시간과 무관하게 1초 이내 즉시 뚫립니다.",
        [
            ("Password Metrics Dataset (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset"),
            ("Password Strength Dataset (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords")
        ]
    ),
    "t1": (
        "Password Metrics Dataset (Kaggle) — 고유 25,000건 구조 피처",
        "원본 100,000행 중 중복을 제거한 고유 25,000개 데이터입니다(개인정보 보호를 위해 비밀번호 원문은 제거하고 길이·문자 종류·패턴 등 20개 수치 피처만 기록). "
        "'버팀 시간'은 문자 종류 크기(N)와 길이(L)에 따른 경우의 수(N^L)를 처음부터 끝까지 대입(Brute-force)할 때 걸리는 이론적 연산 시간이며, 극단값 왜곡을 막기 위해 로그 스케일과 중앙값(Median)으로 집계했습니다. "
        "[한계] 5개 강도 등급별로 약 5,000건씩 추출한 균등 표본이므로 전체 인터넷 이용자 분포를 대표하지 않으며, 키보드 연속 입력·생일 패턴은 별도 감점 없이는 과대평가될 수 있습니다.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset")]
    ),
    "t2": (
        "Password Strength and Vulnerability Dataset (Kaggle / Information is Beautiful) — 상위 500건",
        "전 세계에서 가장 흔히 쓰이는 비밀번호 상위 500개 공개 데이터입니다. "
        "① 로그인 창 찍기(online_sec): 웹사이트 로그인 화면에서 직접 입력하는 공격(서버 지연으로 초당 약 10회 대입 가정), "
        "② 서버 유출 파일 돌리기(offline_crack_sec): 해킹으로 유출된 해시 파일을 고성능 장비로 대입하는 공격(초당 약 10억 회 가정, 온라인보다 정확히 10^8배=1억 배 빠름)을 초 단위로 환산한 값입니다. "
        "[한계] 표기된 시간은 사전 없이 'aaaa'부터 순서대로 넣어보는 순수 무차별 대입 기준이므로, 실제 1순위 공격인 '사전 대입 공격(Dictionary Attack)' 적용 시 500개 모두 1초 안에 즉시 해독됩니다.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords")]
    ),
    "t3": (
        "자체 학습 예측 모델(Random Forest) · 조건 추가 실험(What-If) · HIBP 실시간 유출 조회(k-익명성)",
        "고유 25,000건의 20개 피처로 학습한 Random Forest 회귀 모델에 키보드 연속·반복·연도 패턴 감점을 결합해 예상 버팀 시간을 산출하고, 상위 500개 흔한 목록 및 HIBP(Have I Been Pwned) API로 실제 유출 여부를 함께 점검합니다. "
        "입력한 비밀번호는 외부 서버로 전송되거나 파일로 저장되지 않으며, HIBP 조회 시에도 SHA-1 암호문 40자리 중 앞 5자리만 보내 수백 개의 후보 묶음을 받아 내 컴퓨터 안에서만 비교하는 k-익명성(k-Anonymity) 방식을 사용합니다. "
        "유출 횟수는 HIBP 응답값 그대로만 표시하며 버팀 시간 계산과 섞지 않고, 네트워크 오류 시 '안전'이 아닌 '확인 불가'로 분리 표기합니다.",
        [
            ("학습 데이터: Password Metrics Dataset (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset"),
            ("흔한 목록: Password Strength Dataset (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords"),
            ("유출 조회 API: HIBP Pwned Passwords", "https://haveibeenpwned.com/API/v3#PwnedPasswords")
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
RE_MMDD = re.compile(r"(?<!\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
RE_YYMMDD = re.compile(r"(?<!\d)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
RE_LEET = re.compile(r"(?<=[a-z])[@$013457]+(?=[a-z])")

_BTN_W = dict(width="stretch") if "width" in inspect.signature(st.button).parameters else dict(use_container_width=True)
_POP_W = dict(width="stretch") if "width" in inspect.signature(st.popover).parameters else dict(use_container_width=True)

@st.cache_data(ttl=3600, show_spinner=False)
def _hibp_range(prefix: str) -> str:
    req = urllib.request.Request(
        f"https://api.pwnedpasswords.com/range/{prefix}",
        headers={"User-Agent": "PasswordSecurityDashboard", "Add-Padding": "true"}
    )
    with urllib.request.urlopen(req, timeout=2.8) as res:
        return res.read().decode("utf-8")

def check_hibp(pw: str):
    sha1 = hashlib.sha1(pw.encode("utf-8")).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]
    try:
        body = _hibp_range(prefix)
    except Exception:
        return None
    for line in body.splitlines():
        h, _, cnt = line.partition(":")
        if h.strip() == suffix:
            try:
                return int(cnt)
            except ValueError:
                return None
    return 0

def _longest_seq(low: str) -> int:
    best = run = 1
    for a, b in zip(low, low[1:]):
        if any(a in r and b in r and abs(r.index(b) - r.index(a)) == 1 for r in SEQ_ROWS):
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best

def extract_features(pw: str) -> dict:
    L = max(len(pw), 1)
    low = pw.lower()
    n_lower = sum(c.islower() and c.isascii() for c in pw)
    n_upper = sum(c.isupper() and c.isascii() for c in pw)
    n_digit = sum(c.isdigit() and c.isascii() for c in pw)
    n_non = sum(not c.isascii() for c in pw)
    n_special = len(pw) - n_lower - n_upper - n_digit - n_non
    cnt = {}
    for c in pw:
        cnt[c] = cnt.get(c, 0) + 1
    return {
        "length": len(pw), "n_lower": n_lower, "n_upper": n_upper, "n_digit": n_digit,
        "n_special": n_special, "n_non": n_non,
        "r_lower": n_lower / L, "r_upper": n_upper / L, "r_digit": n_digit / L, "r_special": n_special / L,
        "n_classes": int(n_lower > 0) + int(n_upper > 0) + int(n_digit > 0) + int(n_special > 0) + int(n_non > 0),
        "unique_ratio": len(cnt) / L,
        "shannon": -sum(v / L * math.log2(v / L) for v in cnt.values()),
        "longest_seq": _longest_seq(low),
        "longest_rep": max((len(m.group(0)) for m in re.finditer(r"(.)\1*", pw)), default=1),
        "has_year": int(bool(re.search(r"(19|20)\d\d", pw))),
        "has_date": int(bool(RE_MMDD.search(pw) or RE_YYMMDD.search(pw))),
        "has_leet": int(bool(RE_LEET.search(low))),
        "digit_only": int(pw.isdigit()),
        "alpha_only": int(pw.isalpha())
    }

def md(s, gap=0):
    tail = f'<div style="height:{gap}px"></div>' if gap > 0 else ""
    st.markdown("".join(l.strip() for l in s.splitlines()) + tail, unsafe_allow_html=True)

def sup(n):
    return str(int(n)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))

def human(s):
    """누구나 직관적으로 읽을 수 있는 자연스러운 한국어 시간 표기 (밀리초·지수 표기 제거)"""
    if s is None or not np.isfinite(s):
        return "측정 불가"
    if s < 0.01:
        return "0.01초 미만"
    if s < 1:
        return f"{s:.1f}초" if s >= 0.1 else "0.1초 미만"
    for lim, div, u in [(60, 1, "초"), (3600, 60, "분"), (86400, 3600, "시간"), (2629800, 86400, "일"), (YR, 2629800, "개월")]:
        if s < lim:
            return f"{s / div:.0f}{u}"
    y = s / YR
    if y < 1e4:
        return f"{y:,.0f}년"
    if y < 1e8:
        return f"{y / 1e4:,.0f}만 년"
    return "1억 년 이상"

def axis_human(s):
    """차트 축 눈금용 짧은 시간 표기"""
    if s < 0.05:
        return "0.01초"
    if s < 1:
        return "0.1초"
    for lim, div, u in [(60, 1, "초"), (3600, 60, "분"), (86400, 3600, "시간"), (2629800, 86400, "일"), (YR, 2629800, "개월")]:
        if s < lim:
            return f"{s / div:.0f}{u}"
    y = s / YR
    if y < 1e4:
        return f"{y:,.0f}년"
    if y < 1e8:
        return f"{y / 1e4:.0f}만 년"
    if y < 1e12:
        return f"{y / 1e8:.0f}억 년"
    if y < 1e16:
        return f"{y / 1e12:.0f}조 년"
    return f"10{sup(round(math.log10(y)))}년"

def rank_text(pct: float) -> str:
    """상위/하위 혼동 없이 바로 이해되는 상대 순위 표기"""
    top = max(0.1, 100.0 - pct)
    if top <= 50.0:
        return f"상위 {top:.0f}%" if top >= 1 else f"상위 {top:.1f}%"
    bot = max(1.0, pct)
    return f"하위 {bot:.0f}%"

def mask_pw(pw: str) -> str:
    if len(pw) <= 2:
        return "•" * len(pw)
    return pw[0] + "•" * min(len(pw) - 2, 10) + pw[-1]

def combo_label(feats_or_sc: dict) -> str:
    if feats_or_sc.get("n_lo", 0) == 0 and feats_or_sc.get("n_up", 0) == 0 and feats_or_sc.get("n_sp", 0) == 0 and feats_or_sc.get("n_dg", 0) > 0:
        return "숫자만"
    if feats_or_sc.get("n_dg", 0) == 0 and feats_or_sc.get("n_sp", 0) == 0:
        return "영문만"
    if feats_or_sc.get("n_sp", 0) > 0:
        return "특수문자 포함"
    return "영문+숫자"

def tick_axis(fig, lo, hi, axis="y"):
    lo, hi = float(lo), float(hi)
    span = max(hi - lo, 1.0)
    keep = []
    for v in CAND:
        x = math.log10(v)
        if lo - 0.35 <= x <= hi + 0.35 and (not keep or x - keep[-1] >= span / 5.5):
            keep.append(x)
    kw = dict(tickvals=keep, ticktext=[axis_human(10 ** x) for x in keep], range=[lo - 0.45, hi + 0.45], tickangle=0)
    if axis == "y":
        fig.update_yaxes(**kw)
    else:
        fig.update_xaxes(**kw)

def style(fig, h=300, t=20, l=12, r=18, b=12):
    fig.update_layout(
        height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=12.5, color="#1e293b"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)),
        hoverlabel=dict(font_family=FONT, bgcolor="white", font_size=12.5)
    )
    fig.update_xaxes(gridcolor="#eef2f6", zeroline=False, showline=False, automargin=True)
    fig.update_yaxes(gridcolor="#eef2f6", zeroline=False, showline=False, automargin=True)
    return fig

_PW = "width" in inspect.signature(st.plotly_chart).parameters
def show(fig, key):
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key, **(dict(width="stretch") if _PW else dict(use_container_width=True)))

def card_head(title: str, sub: str, table_fn=None):
    """흰색 통합 카드 내부 상단 제목 + 쉬운 1줄 설명"""
    if table_fn is None:
        md(f'<div class="ch"><div class="ct">{title}</div><div class="cs">{sub}</div></div>')
    else:
        c1, c2 = st.columns([8.0, 2.0], vertical_alignment="top")
        with c1:
            md(f'<div class="ch" style="margin-bottom:2px"><div class="ct">{title}</div></div>')
        with c2:
            with st.popover("📋 표 보기", **_POP_W):
                table_fn()
        md(f'<div class="ch"><div class="cs" style="margin-top:0">{sub}</div></div>')

def summary_cards(items):
    """답답한 원색 블록을 대체하는 깔끔한 화이트 요약 카드 4개"""
    bgs = [("#eff6ff", "#2563eb"), ("#fef2f2", "#dc2626"), ("#fffbeb", "#d97706"), ("#ecfdf5", "#059669")]
    html_cards = []
    for i, (icon, label, val, desc) in enumerate(items):
        bg, fg = bgs[i % len(bgs)]
        html_cards.append(
            f'<div class="scard">'
            f'<div class="sicon" style="background:{bg};color:{fg}">{icon}</div>'
            f'<div class="sbody">'
            f'<div class="slbl">{label}</div>'
            f'<div class="sval">{val}</div>'
            f'<div class="sdsc">{desc}</div>'
            f'</div></div>'
        )
    md(f'<div class="sgrid">{"".join(html_cards)}</div>')

def source(key):
    n, t, links = SRC[key]
    with st.expander("📚 데이터 출처 · 시간(초) 계산 수식(Nᴸ) · 표본 한계점 (검증 기준)"):
        link_md = " &nbsp;|&nbsp; ".join(f"🔗 [{label}]({url})" for label, url in links)
        st.markdown(f"**{n}**\n\n{t}\n\n{link_md}")

def tile_grid(qq):
    lo, hi = 0.0, 7.5
    A, B, Cc = (254, 202, 202), (254, 240, 188), (187, 247, 208)
    def col(v):
        vc = float(np.clip(v, lo, hi))
        t = (vc - lo) / (hi - lo)
        p, q_, u = (A, B, t * 2) if t < 0.5 else (B, Cc, (t - 0.5) * 2)
        return "rgb(%d,%d,%d)" % tuple(int(p[k] + (q_[k] - p[k]) * u) for k in range(3))
    cells = "".join(
        f'<div class="tl" style="background:{col(v)}" '
        f'title="{_html.escape(str(w))} ({_html.escape(str(ck))}) · 로그인 창 대입: {human(s_)} / 유출 파일 대입: {human(of_)} / 사전 공격 시 1초 내 해독">'
        f'<span class="rk">{int(r)}위 · {_html.escape(str(ck))}</span>'
        f'<b>{_html.escape(str(w))}</b>'
        f'<span class="tm">로그인 창 {human(s_)}</span></div>'
        for r, w, ck, v, s_, of_ in zip(qq["rank"], qq.password, qq.cat_ko, qq.lon, qq.online_sec, qq.offline_crack_sec)
    )
    md(f'<div class="tg">{cells}</div>')

def _find_file(name):
    for base in [D, Path(__file__).parent]:
        if (base / name).exists():
            return base / name
    return D / name

@st.cache_data
def load():
    m = pd.read_csv(_find_file("features_unique.csv"))
    m["등급"] = pd.Categorical(m.class_strength.map(KO), ORDER, ordered=True)
    m["lc"] = np.log10(m.crack_time_sec.clip(lower=1e-12))
    conds = [m["digit_only"] == 1, m["alpha_only"] == 1, m["n_special"] > 0]
    choices = ["숫자만", "영문만", "특수문자 포함"]
    m["조합유형"] = np.select(conds, choices, default="영문+숫자")

    p = pd.read_csv(_find_file("common_passwords.csv"))
    p["lon"] = np.log10(p.online_sec.clip(lower=1e-6))
    p["lof"] = np.log10(p.offline_crack_sec.clip(lower=1e-6))
    p["cat_ko"] = p["category"].astype(str).str.strip().map(lambda x: CAT_KO.get(x, x))
    return m, p

@st.cache_resource
def train_model(_m: pd.DataFrame):
    shuffled = _m.sample(frac=1.0, random_state=42)
    X = shuffled[FEAT_COLS]
    y = shuffled["lc"]
    model = RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

M, P = load()
ML_MODEL = train_model(M)
TOP = set(P.password.astype(str).str.lower())

md("""<style>@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html,body,[class*="css"],.stApp{font-family:Pretendard,-apple-system,'Apple SD Gothic Neo',sans-serif!important;overflow-x:hidden;color-scheme:light}
.stApp{background:#f4f6fa}#MainMenu,footer{visibility:hidden}[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}

/* 전체 여백 대폭 확대 (답답함 해소) */
.block-container{max-width:1460px;width:100%;box-sizing:border-box;padding:1.4rem 2.2rem 2.8rem;position:relative;z-index:1;animation:up .45s ease both}
.block-container>[data-testid="stVerticalBlock"]{gap:1.15rem!important}
@keyframes up{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}

/* 상단 네비게이션 헤더 */
.st-key-hdr{background:linear-gradient(100deg,#111827,#1e293b);border-radius:22px;padding:12px 22px;border:1px solid #334155;box-shadow:0 8px 24px rgba(15,23,42,.12)}
.st-key-hdr [data-testid="stHorizontalBlock"]{align-items:center!important;min-height:46px}
.st-key-hdr [data-testid="stColumn"]{display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stElementContainer"],.st-key-hdr [data-testid="stMarkdownContainer"],.st-key-hdr [data-testid="stButton"],.st-key-hdr [data-testid="stButtonGroup"]{margin:0!important;padding:0!important;display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stMarkdownContainer"] p{margin:0!important;padding:0!important;line-height:46px!important}
div.st-key-hdr,div.st-key-hdr *,div.st-key-hdr p,div.st-key-hdr button *{color:#f8fafc!important}
.ht{font-size:19px;font-weight:800;letter-spacing:-.02em;white-space:nowrap;display:inline-flex;align-items:center;gap:8px;height:46px;line-height:46px}
.st-key-hdr [data-testid="stButtonGroup"]{flex-wrap:nowrap;gap:8px;height:46px}
.st-key-hdr [data-testid="stButtonGroup"] button{height:44px!important;min-height:44px!important;padding:0 16px!important;font-size:14.2px;font-weight:700;border-radius:13px;background:#1e293b;border:1px solid #334155;white-space:nowrap;transition:all .2s;display:inline-flex;align-items:center;justify-content:center}
.st-key-hdr [data-testid="stButtonGroup"] button:hover{border-color:#60a5fa;transform:translateY(-1px)}
.st-key-hdr [data-testid="stButtonGroup"] button[aria-checked="true"],.st-key-hdr [data-testid="stButtonGroup"] button[aria-pressed="true"],.st-key-hdr button[data-testid="stBaseButton-segmented_controlActive"]{background:#3b82f6!important;border-color:#93c5fd!important;box-shadow:0 4px 14px rgba(59,130,246,.45)}
.st-key-hdr button[kind="secondary"]{background:#1e293b;border:1px solid #334155;border-radius:13px;height:44px!important;min-height:44px!important;padding:0 14px!important}

/* 통합 화이트 카드 (제목 + 설명 + 차트가 한 카드 안에서 여유 있게 배치됨) */
div[class*="st-key-ucard_"]{background:#ffffff;border-radius:22px;padding:22px 24px 18px!important;border:1px solid #e5e9f2;box-shadow:0 4px 20px rgba(15,23,42,.035)}
div[class*="st-key-ucard_"] div[data-testid="stPlotlyChart"]{box-shadow:none!important;border-radius:12px;background:#fff}

/* 페이지 상단 질문 배너 */
.pbanner{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;padding:4px 2px 2px}
.pstep{display:inline-block;font-size:12px;font-weight:800;color:#2563eb;background:#dbeafe;padding:4px 10px;border-radius:99px;margin-bottom:8px}
.ptitle{font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-.025em;line-height:1.3}
.psub{font-size:14px;color:#475569;margin-top:6px;line-height:1.55;word-break:keep-all}
.sync-pill{display:inline-flex;align-items:center;gap:6px;background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;font-size:12.5px;font-weight:700;padding:7px 13px;border-radius:12px;white-space:nowrap}

/* 카드 내부 헤더 */
.ch{margin-bottom:10px}
.ct{font-size:16.5px;font-weight:800;color:#0f172a;letter-spacing:-.02em;line-height:1.35}
.cs{font-size:13px;color:#64748b;margin-top:4px;line-height:1.5;word-break:keep-all}

/* 화이트 요약 KPI 카드 4개 */
.sgrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
.scard{background:#ffffff;border-radius:20px;padding:18px 20px;border:1px solid #e5e9f2;box-shadow:0 4px 18px rgba(15,23,42,.035);display:flex;align-items:center;gap:15px;min-width:0}
.sicon{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:23px;flex:none}
.sbody{min-width:0;flex:1}
.slbl{font-size:12.5px;font-weight:700;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sval{font-size:24px;font-weight:800;color:#0f172a;letter-spacing:-.02em;margin-top:2px;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sdsc{font-size:11.8px;color:#64748b;margin-top:4px;line-height:1.35;word-break:keep-all}

/* 1번 탭: 내 비밀번호 구성 조건 숫자 표시 바 */
.cond-wrap{background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.pw-chip-group{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.pw-tag{font-size:12px;font-weight:800;padding:5px 10px;border-radius:8px;background:#dbeafe;color:#1d4ed8}
.pw-tag.direct{background:#e0e7ff;color:#3730a3}
.pw-mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;font-weight:800;color:#0f172a;background:#fff;padding:5px 13px;border-radius:9px;border:1px solid #cbd5e1;letter-spacing:.03em}
.cond-pills{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.cpill{background:#ffffff;border:1px solid #dce3ee;border-radius:10px;padding:6px 11px;font-size:12.5px;font-weight:700;color:#334155;display:inline-flex;align-items:center;gap:5px}
.cpill b{color:#0f172a;font-size:13.5px;font-weight:800}
.cpill.up-chg{background:#eff6ff;border-color:#93c5fd;color:#1d4ed8}
.cpill.up-chg b{color:#1d4ed8}
.cpill.warn-p{background:#fef2f2;border-color:#fecaca;color:#dc2626}
.cpill.ok-p{background:#ecfdf5;border-color:#a7f3d0;color:#059669}

/* 입력 바 & What-If 버튼 수평·높이 정렬 */
.st-key-ucard_t1_ctrl [data-testid="stForm"]{margin:0!important;padding:0!important;border:0!important}
.wi-row-lbl{font-size:13.5px;font-weight:800;color:#0f172a;white-space:nowrap;height:42px;line-height:42px;display:inline-flex;align-items:center;gap:6px}
.st-key-wi_bar [data-testid="stHorizontalBlock"]{align-items:center!important;min-height:42px}
.st-key-wi_bar [data-testid="stColumn"],.st-key-wi_bar [data-testid="stElementContainer"],.st-key-wi_bar [data-testid="stMarkdownContainer"],.st-key-wi_bar [data-testid="stButton"]{display:flex!important;align-items:center!important;margin:0!important;padding:0!important;width:100%!important}
.st-key-wi_bar button{width:100%!important;height:42px!important;min-height:42px!important;border-radius:12px!important;font-size:13px!important;font-weight:700!important;transition:all .2s}
.st-key-wi_bar button[kind="primary"]{background:#3b82f6!important;border:1px solid #2563eb!important;box-shadow:0 4px 12px rgba(59,130,246,.3)!important}
.st-key-wi_bar button[kind="primary"],.st-key-wi_bar button[kind="primary"] *{color:#ffffff!important;font-weight:800!important}
.st-key-wi_bar button[kind="secondary"]{background:#f8fafc!important;border:1px solid #cbd5e1!important}
.st-key-wi_bar button[kind="secondary"] *{color:#1e293b!important;font-weight:700!important}

.st-key-d_ex0 button,.st-key-d_ex1 button,.st-key-d_ex2 button,.st-key-d_ex3 button{width:100%;height:42px!important;min-height:42px!important;border-radius:12px;border:1px solid #e2e8f0;background:#f8fafc;transition:all .18s;white-space:nowrap}
.st-key-d_ex0 button:hover,.st-key-d_ex1 button:hover,.st-key-d_ex2 button:hover,.st-key-d_ex3 button:hover{background:#eff6ff;border-color:#93c5fd;transform:translateY(-1px)}
.st-key-d_ex0 button *,.st-key-d_ex1 button *,.st-key-d_ex2 button *,.st-key-d_ex3 button *{color:#1e293b!important;font-size:12.5px;font-weight:700}

[data-testid="stForm"] button{background:#2563eb;border:0;border-radius:12px;height:42px!important;min-height:42px!important;width:100%}[data-testid="stForm"] button *{color:#fff!important;font-weight:700;font-size:13.5px}[data-testid="stForm"] input{border-radius:12px;min-height:42px}[data-testid="InputInstructions"]{display:none!important}

/* 1번 탭 우측 세부 진단 카드 내부 */
.grade-pill{font-size:12.5px;font-weight:800;padding:5px 12px;border-radius:99px;color:#fff!important}
.stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:10px 0 12px}
.stat-box{background:#f8fafc;border-radius:15px;padding:13px 15px;border:1px solid #e2e8f0}
.stat-lbl{font-size:12px;font-weight:700;color:#64748b;margin-bottom:4px}
.stat-val{font-size:21px;font-weight:800;color:#0f172a;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-sub{font-size:11.5px;color:#64748b;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hibp-bar{border-radius:12px;padding:10px 14px;margin-bottom:8px;font-size:12.8px;font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:8px}
.hibp-bad{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca}
.hibp-ok{background:#ecfdf5;color:#047857;border:1px solid #a7f3d0}
.hibp-na{background:#f8fafc;color:#475569;border:1px solid #e2e8f0}
.chk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:8px}
.chip{display:flex;align-items:center;justify-content:center;border-radius:10px;padding:8.5px 6px;font-size:12px;font-weight:700;text-align:center;white-space:nowrap}.ok{background:#ecfdf5;color:#047857;border:1px solid #a7f3d0}.no{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca}.na{background:#f1f5f9;color:#475569;border:1px solid #cbd5e1}

/* 2번 탭 하단: 해커의 4단계 공격 순서 (가로 4열 카드) */
.atk-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:6px}
.atk-step{background:#f8fafc;border:1px solid #e2e8f0;border-top:4px solid #3b82f6;border-radius:16px;padding:16px 16px 15px;display:flex;flex-direction:column;gap:8px}
.atk-step.s1{border-top-color:#ef4444;background:#fffafa}
.atk-step.s2{border-top-color:#f59e0b;background:#fffdf7}
.atk-step.s3{border-top-color:#3b82f6}
.atk-step.s4{border-top-color:#10b981;background:#f7fdf9}
.atk-badge{display:inline-flex;align-items:center;justify-content:space-between;gap:6px}
.atk-num{font-size:11.5px;font-weight:800;padding:3px 9px;border-radius:99px;background:#e2e8f0;color:#1e293b}
.atk-step.s1 .atk-num{background:#fee2e2;color:#b91c1c}
.atk-step.s2 .atk-num{background:#fef3c7;color:#b45309}
.atk-step.s3 .atk-num{background:#dbeafe;color:#1d4ed8}
.atk-step.s4 .atk-num{background:#d1fae5;color:#047857}
.atk-tech{font-size:11.5px;font-weight:700;color:#64748b}
.atk-ttl{font-size:15px;font-weight:800;color:#0f172a;line-height:1.35;word-break:keep-all}
.atk-txt{font-size:12.8px;color:#475569;line-height:1.55;word-break:keep-all}

/* 4번 탭 타일 지도 */
.tg{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;height:315px;overflow-y:auto;padding:4px 4px 4px 0;box-sizing:border-box;align-content:start}
.tg::-webkit-scrollbar{width:6px}.tg::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:6px}
.tl{border-radius:14px;padding:11px 8px;text-align:center;min-width:0;display:flex;flex-direction:column;gap:4px;justify-content:center;border:1px solid rgba(15,23,42,.06);transition:transform .2s}
.tl:hover{transform:translateY(-2px)}
.tl b{font-size:14.5px;color:#0f172a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rk{font-size:11px;color:#334155;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tm{font-size:11.5px;font-weight:700;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* 라이트모드 텍스트 가독성 고정 */
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] strong,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *,[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"],[data-testid="stToggle"] label *{color:#0f172a!important}
span[data-baseweb="tag"]{background:#3b82f6!important;border-radius:8px}[role="slider"]{background:#3b82f6!important}[data-testid="stExpander"]{border-radius:16px;background:#fff;border:1px solid #e5e9f2}
[data-testid="stPopover"] button{border-radius:99px!important;padding:3px 12px!important;min-height:32px!important;font-size:12px!important;font-weight:700!important;background:#f8fafc!important;border:1px solid #cbd5e1!important}
[data-testid="stPopover"] button:hover{border-color:#3b82f6!important;color:#2563eb!important}

.spl{position:fixed;inset:0;z-index:99999;background:#0f172a;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.2s ease forwards}
.spl div{color:#fff;font-size:clamp(28px,5.5vw,58px);font-weight:800;letter-spacing:-.03em;animation:si 1.2s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.94);filter:blur(8px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,70%{opacity:1}100%{opacity:0;visibility:hidden}}</style>""")

if not st.session_state.get("splash"):
    st.session_state["splash"] = True
    md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>')

NAMES = [
    "🔍 1. 내 비밀번호 진단",
    "⚡ 2. 해킹 방법과 속도",
    "📏 3. 길고 복잡할수록 안전할까?",
    "🚨 4. 흔한 비밀번호의 함정"
]
if "nav" not in st.session_state:
    st.session_state["nav"] = NAMES[0]

def _keep():
    if st.session_state.get("nav") is None:
        st.session_state["nav"] = st.session_state.get("last_nav", NAMES[0])

with st.container(key="hdr"):
    hc = st.columns([2.4, 8.4, 1.2], vertical_alignment="center")
    hc[0].markdown('<div class="ht">🔐 비밀번호 보안 대시보드</div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        hc[1].segmented_control("메뉴", NAMES, key="nav", label_visibility="collapsed", on_change=_keep)
    else:
        hc[1].radio("메뉴", NAMES, horizontal=True, label_visibility="collapsed", key="nav")
    hc[2].button("▶ 인트로", on_click=lambda: st.session_state.update(splash=False), **_BTN_W)

cur = st.session_state.get("nav") or NAMES[0]
st.session_state["last_nav"] = cur
page = NAMES.index(cur)

def _setall(key, vals):
    st.session_state[key] = list(vals)

def multi_pick(col, label, options, key):
    if key not in st.session_state:
        st.session_state[key] = list(options)
    with col.popover(f"{label} ({len(st.session_state[key])}/{len(options)}개 선택)", **_POP_W):
        b = st.columns(2)
        b[0].button("전체 선택", key=f"{key}_all", on_click=_setall, args=(key, options), **_BTN_W)
        b[1].button("전체 해제", key=f"{key}_none", on_click=_setall, args=(key, []), **_BTN_W)
        if hasattr(st, "pills"):
            return st.pills(label, options, selection_mode="multi", key=key, label_visibility="collapsed") or []
        return st.multiselect(label, options, key=key, label_visibility="collapsed")

def _eval_struct(pw: str) -> dict:
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
    raw_sec = 10 ** adj_lc
    pct = float((M["lc"] <= adj_lc).mean() * 100)
    score = float(np.clip((adj_lc + 2.0) / 14.0 * 100.0, 0, 100))
    common = low in TOP
    n_sp_all = feats["n_special"] + feats["n_non"]
    return dict(
        L=L, adj_lc=adj_lc, raw_sec=raw_sec, pct=pct, score=score, common=common,
        n_lo=feats["n_lower"], n_up=feats["n_upper"], n_dg=feats["n_digit"], n_sp=n_sp_all,
        seq=seq, rep=rep, yr=yr,
        up=feats["n_upper"] > 0, dg=feats["n_digit"] > 0, sp=n_sp_all > 0
    )

def diagnose(pw: str, example_label: str = None):
    is_ex = example_label is not None
    base = _eval_struct(pw)
    hibp_cnt = check_hibp(pw)
    leaked = hibp_cnt is not None and hibp_cnt > 0

    sp_ch = "!" if not pw.endswith("!") else "#"
    pw_sp = pw + sp_ch
    if pw.isdigit():
        add4 = "7392"
    elif pw.isalpha():
        add4 = "mpxq" if pw.islower() else "MpXq"
    else:
        add4 = "m7p9"
    pw_l4 = pw + add4
    pw_both = pw + ("!K9m" if not pw.endswith("!") else "#K9m")

    s_sp = _eval_struct(pw_sp)
    s_l4 = _eval_struct(pw_l4)
    s_both = _eval_struct(pw_both)

    s_sp["adj_lc"] = max(s_sp["adj_lc"], base["adj_lc"] + (1.2 if not base["sp"] else 0.7))
    s_sp["raw_sec"] = 10 ** s_sp["adj_lc"]
    s_sp["pct"] = float((M["lc"] <= s_sp["adj_lc"]).mean() * 100)
    s_sp["score"] = float(np.clip((s_sp["adj_lc"] + 2.0) / 14.0 * 100.0, 0, 100))

    s_l4["adj_lc"] = max(s_l4["adj_lc"], base["adj_lc"] + 2.2)
    s_l4["raw_sec"] = 10 ** s_l4["adj_lc"]
    s_l4["pct"] = float((M["lc"] <= s_l4["adj_lc"]).mean() * 100)
    s_l4["score"] = float(np.clip((s_l4["adj_lc"] + 2.0) / 14.0 * 100.0, 0, 100))

    s_both["adj_lc"] = max(s_both["adj_lc"], max(s_sp["adj_lc"], s_l4["adj_lc"]) + 1.4)
    s_both["raw_sec"] = 10 ** s_both["adj_lc"]
    s_both["pct"] = float((M["lc"] <= s_both["adj_lc"]).mean() * 100)
    s_both["score"] = float(np.clip((s_both["adj_lc"] + 2.0) / 14.0 * 100.0, 0, 100))

    def _pack(st_dict, key, label, short, raw_str, mask_str, apply_leak=False):
        sec = min(st_dict["raw_sec"], 0.005) if st_dict["common"] else st_dict["raw_sec"]
        sc = st_dict["score"]
        if st_dict["common"]:
            sc = min(sc, 5.0)
        elif apply_leak and leaked:
            sc = min(sc, 10.0 if hibp_cnt < 100 else 5.0)
        return dict(
            key=key, label=label, short=short,
            plain=raw_str, masked=mask_str,
            L=st_dict["L"], sec=sec, raw_sec=st_dict["raw_sec"],
            pct=st_dict["pct"], score=sc, common=st_dict["common"],
            n_lo=st_dict["n_lo"], n_up=st_dict["n_up"], n_dg=st_dict["n_dg"], n_sp=st_dict["n_sp"],
            seq=st_dict["seq"], rep=st_dict["rep"], yr=st_dict["yr"],
            up=st_dict["up"], dg=st_dict["dg"], sp=st_dict["sp"]
        )

    m_base = mask_pw(pw)
    scenarios = {
        "base": _pack(base, "base", "현재 비밀번호", "현재 상태", pw, m_base, apply_leak=True),
        "sp1": _pack(s_sp, "sp1", "특수문자 1개 추가", "+특수문자 1개", pw_sp, m_base + sp_ch, apply_leak=False),
        "len4": _pack(s_l4, "len4", "길이 4글자 추가", "+길이 4글자", pw_l4, m_base + add4, apply_leak=False),
        "both": _pack(s_both, "both", "특수문자 + 4글자 추가", "+특수·4글자", pw_both, m_base + ("!K9m" if not pw.endswith("!") else "#K9m"), apply_leak=False)
    }

    st.session_state["diag"] = dict(
        hibp=hibp_cnt, is_example=is_ex, example_label=example_label,
        base_common=base["common"], combo=combo_label(base), scenarios=scenarios
    )
    st.session_state["wi_mode"] = "base"
    # 2번 탭의 비교 글자 수 슬라이더도 내 비밀번호 길이에 맞춰 자동 연동
    st.session_state["i_l"] = int(np.clip(base["L"], 4, 24))

if "diag" not in st.session_state:
    diagnose("latte79", example_label="짧은 일상형")
if "wi_mode" not in st.session_state:
    st.session_state["wi_mode"] = "base"

# =========================================================
# TAB 1. 내 비밀번호 진단 & 글자 추가 실험
# =========================================================
def p_lab():
    md("""<div class="pbanner">
        <div>
            <span class="pstep">STEP 1 · 내 비밀번호 진단 & 조건 실험</span>
            <div class="ptitle">내 비밀번호는 얼마나 버틸까? 글자를 더하면 어떻게 바뀔까?</div>
            <div class="psub">비밀번호에 들어간 글자 종류·개수를 확인하고, 특수문자나 길이를 더했을 때 버팀 시간이 몇 배 늘어나는지 바로 비교해 보세요.</div>
        </div>
    </div>""")

    with st.container(key="ucard_t1_ctrl"):
        r1 = st.columns([4.4, 1.4, 1.05, 1.05, 1.05, 1.05], gap="small", vertical_alignment="center")
        with r1[0]:
            with st.form("f_pw", clear_on_submit=False, border=False):
                fc = st.columns([3.5, 1.3], gap="small", vertical_alignment="center")
                pw_in = fc[0].text_input("비밀번호 입력", type="password", placeholder="진단할 비밀번호를 입력하세요 (외부 전송 안 됨)", label_visibility="collapsed")
                if fc[1].form_submit_button("진단하기") and pw_in:
                    diagnose(pw_in, example_label=None)
        show_pw = r1[1].toggle("👁️ 글자 보기", value=False, key="show_pw")
        examples = [
            ("🔓 흔한 단어", "흔한 단어형", "password123"),
            ("☕ 짧은 일상", "짧은 일상형", "latte79"),
            ("📅 연도·패턴", "연도·패턴형", "Summer2024!"),
            ("🛡️ 무작위 안전", "무작위 안전형", "vK9#mQ2$zL7@pW4xR8")
        ]
        for i, (col, (btn_lab, ex_name, ex_val)) in enumerate(zip(r1[2:], examples)):
            if col.button(btn_lab, key=f"d_ex{i}", **_BTN_W):
                diagnose(ex_val, example_label=ex_name)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        d = st.session_state["diag"]
        scs = d["scenarios"]
        wi_key = st.session_state.get("wi_mode", "base")
        if wi_key not in scs:
            wi_key = "base"
        base_sc = scs["base"]
        act_sc = scs[wi_key]
        hibp_cnt = d.get("hibp")
        leaked = hibp_cnt is not None and hibp_cnt > 0

        # 내 비밀번호 표시 문자열 & 실제 들어간 조건 개수 표시 (한눈에 확인!)
        is_ex = d.get("is_example", False)
        if is_ex or show_pw:
            disp_pw = f"{base_sc['plain']}  →  {act_sc['plain']}" if wi_key != "base" else base_sc["plain"]
        else:
            disp_pw = f"{base_sc['masked']}  →  {act_sc['masked']}" if wi_key != "base" else base_sc["masked"]

        src_badge = f'<span class="pw-tag">예시: {_html.escape(str(d["example_label"]))}</span>' if is_ex else '<span class="pw-tag direct">🔒 내가 입력한 비밀번호</span>'

        def _cond_pill(name, b_val, a_val, unit="개"):
            if a_val != b_val:
                diff = a_val - b_val
                return f'<span class="cpill up-chg">{name} <b>{b_val} → {a_val}{unit} (+{diff})</b></span>'
            return f'<span class="cpill">{name} <b>{a_val}{unit}</b></span>'

        pat_list = []
        if act_sc["seq"]:
            pat_list.append("연속글자(123/qwe)")
        if act_sc["rep"]:
            pat_list.append("반복글자(aaa)")
        if act_sc["yr"]:
            pat_list.append("연도·날짜")
        pat_pill = (
            f'<span class="cpill warn-p">⚠️ 주의 패턴: <b>{", ".join(pat_list)}</b></span>'
            if pat_list else '<span class="cpill ok-p">✓ 뻔한 패턴 없음</span>'
        )

        pills_html = "".join([
            _cond_pill("총 길이", base_sc["L"], act_sc["L"], "자"),
            _cond_pill("소문자", base_sc["n_lo"], act_sc["n_lo"]),
            _cond_pill("대문자", base_sc["n_up"], act_sc["n_up"]),
            _cond_pill("숫자", base_sc["n_dg"], act_sc["n_dg"]),
            _cond_pill("특수문자", base_sc["n_sp"], act_sc["n_sp"]),
            pat_pill
        ])

        md(f"""<div class="cond-wrap">
            <div class="pw-chip-group">{src_badge}<span class="pw-mono">{_html.escape(str(disp_pw))}</span></div>
            <div class="cond-pills">{pills_html}</div>
        </div>""")

        with st.container(key="wi_bar"):
            wc = st.columns([2.2, 1.95, 1.95, 1.95, 1.95], gap="small", vertical_alignment="center")
            wc[0].markdown('<div class="wi-row-lbl">🧪 글자 더해보기 실험</div>', unsafe_allow_html=True)
            wi_opts = [
                ("base", f"현재 상태 ({base_sc['L']}자)"),
                ("sp1", f"+ 특수문자 1개 ({scs['sp1']['L']}자)"),
                ("len4", f"+ 길이 4글자 ({scs['len4']['L']}자)"),
                ("both", f"+ 특수문자 & 4글자 ({scs['both']['L']}자)")
            ]
            for idx, (k_, lbl_) in enumerate(wi_opts):
                if wc[idx + 1].button(lbl_, key=f"wi_b{idx}", type="primary" if wi_key == k_ else "secondary", **_BTN_W):
                    st.session_state["wi_mode"] = k_
                    st.rerun()

    s = act_sc["score"]
    grade, g_col, g_msg = (
        ("위험", RED, "금방 뚫릴 수 있어요") if s < 30 else
        ("주의", ORANGE, "조금 더 보강하면 좋아요") if s < 55 else
        ("양호", BLUE, "일상용으로 꽤 튼튼해요") if s < 80 else
        ("안전", GREEN, "해커가 뚫기 매우 어려워요")
    )
    if wi_key == "base" and (base_sc["common"] or leaked):
        grade, g_col, g_msg = "위험", RED, "이미 알려진 비밀번호예요"

    c1, c2, c3 = st.columns([2.9, 3.55, 3.55], gap="medium")
    with c1:
        with st.container(key="ucard_t1_gauge"):
            card_head("내 비밀번호 안전 점수", g_msg)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(s),
                number=dict(suffix="점", font=dict(size=36, color="#0f172a")),
                title=dict(text=f"등급: {grade}", font=dict(size=15, color=g_col)),
                gauge=dict(
                    axis=dict(range=[0, 100]), bar=dict(color=g_col, thickness=0.34), bgcolor="#f1f5f9", borderwidth=0,
                    steps=[
                        dict(range=[0, 30], color="#fee2e2"),
                        dict(range=[30, 55], color="#fef3c7"),
                        dict(range=[55, 80], color="#dbeafe"),
                        dict(range=[80, 100], color="#d1fae5")
                    ]
                )
            ))
            fig_g.add_annotation(
                text="<b>안전도</b>", x=0.5, y=0.31, xref="paper", yref="paper",
                xanchor="center", yanchor="bottom", showarrow=False,
                font=dict(family=FONT, size=13, color="#64748b")
            )
            show(style(fig_g, 205, t=38, l=34, r=34, b=8), f"d_g_{wi_key}_{round(s)}_{act_sc['L']}")

            vf = go.Figure([
                go.Violin(x=M[M.등급 == g_].length, name=g_, orientation="h", line_color=CCOL[g_], points=False, showlegend=False, spanmode="hard")
                for g_ in ORDER
            ])
            if wi_key != "base" and act_sc["L"] != base_sc["L"]:
                vf.add_vline(x=base_sc["L"], line_color=GRAY, line_dash="dot", annotation_text=f"원래 {base_sc['L']}자", annotation_position="bottom left")
                vf.add_vline(x=act_sc["L"], line_color=BLUE, line_width=2.5, annotation_text=f"변경 {act_sc['L']}자", annotation_position="top right")
            else:
                vf.add_vline(
                    x=act_sc["L"], line_color="#0f172a", line_dash="dash", line_width=2,
                    annotation_text=f"내 길이 {act_sc['L']}자", annotation_position="top right",
                    annotation=dict(bgcolor="rgba(255,255,255,0.88)", borderpad=2)
                )
            vf.update_xaxes(title="실제 25,000개 비밀번호의 등급별 길이 분포(자)")
            show(style(vf, 185, t=18, l=8, r=16, b=10), f"d_vf_{wi_key}_{act_sc['L']}")

    with c2:
        with st.container(key="ucard_t1_wi"):
            order_keys = ["base", "sp1", "len4", "both"]
            y_lbls = [f"{scs[k]['short']} ({scs[k]['L']}자)" for k in order_keys]
            secs = [scs[k]["sec"] for k in order_keys]
            l_vals = [float(np.clip(math.log10(max(v, 1e-3)), -3.0, 18.0)) for v in secs]
            bar_cols = [GREEN if v >= YR else (ORANGE if v >= 3600 else RED) for v in secs]
            b_widths = [3.0 if k == wi_key else 0 for k in order_keys]
            txt_lbl = [human(v) for v in secs]
            t_pairs = [(0.2, "0.01초"), (3.2, "1초"), (6.756, "1시간"), (10.699, "1년"), (12.699, "100년")]
            if max(l_vals) > 12.0:
                t_pairs.append((18.699, "1억 년"))

            card_head(
                "글자를 더했을 때 버팀 시간 변화",
                f"현재 {human(base_sc['sec'])}에서 특수문자+4글자를 더하면 {human(scs['both']['sec'])}까지 늘어납니다."
            )
            fig_wi = go.Figure(go.Bar(
                y=y_lbls, x=[v + 3.2 for v in l_vals], orientation="h",
                marker=dict(color=bar_cols, line=dict(color="#0f172a", width=b_widths)),
                text=txt_lbl, textposition="outside", cliponaxis=False,
                textfont=dict(size=12.5, weight="bold", color="#0f172a"),
                hovertemplate="%{y}: %{text}<extra></extra>"
            ))
            fig_wi.add_vline(x=math.log10(YR) + 3.2, line_dash="dot", line_color="#0f172a", line_width=1.8, annotation_text="안전 기준 (1년)", annotation_position="top")
            fig_wi.update_layout(bargap=0.42)
            fig_wi.update_xaxes(
                title="모든 조합을 대입할 때 걸리는 예상 시간",
                tickvals=[p[0] for p in t_pairs],
                ticktext=[p[1] for p in t_pairs],
                range=[0, max(max(l_vals) + 3.2, math.log10(YR) + 3.2) + 5.8]
            )
            fig_wi.update_yaxes(autorange="reversed", showgrid=False)
            show(style(fig_wi, 390, t=28, l=10, r=32, b=16), f"d_wi_{wi_key}_{act_sc['L']}")

    with c3:
        with st.container(key="ucard_t1_res"):
            if base_sc["common"]:
                common_html = '<div class="hibp-bar hibp-bad"><span>📋 흔한 비밀번호 Top 500</span><span>포함됨 · 1초 내 즉시 뚫림</span></div>'
                common_chip = ("흔한 비번 아님", False)
            else:
                common_html = '<div class="hibp-bar hibp-ok"><span>📋 흔한 비밀번호 Top 500</span><span>안심 (목록에 없음)</span></div>'
                common_chip = ("흔한 비번 아님", True)

            if hibp_cnt is None:
                hibp_html = '<div class="hibp-bar hibp-na"><span>🌐 실제 유출 기록 조회 (HIBP)</span><span>연결 안 됨 · 확인 불가</span></div>'
                hibp_chip = ("유출 확인 불가", None)
            elif hibp_cnt > 0:
                hibp_html = f'<div class="hibp-bar hibp-bad"><span>🚨 실제 유출 기록 (원본 기준)</span><span>전 세계 {hibp_cnt:,}번 유출됨</span></div>'
                hibp_chip = ("유출된 적 없음", False)
            else:
                hibp_html = '<div class="hibp-bar hibp-ok"><span>🛡️ 실제 유출 기록 (원본 기준)</span><span>안심 (유출 0건)</span></div>'
                hibp_chip = ("유출된 적 없음", True)

            chips = [
                ("12글자 이상", act_sc["L"] >= 12), ("대문자 포함", act_sc["up"]), ("숫자 포함", act_sc["dg"]),
                ("특수문자 포함", act_sc["sp"]), common_chip, hibp_chip,
                ("연속글자 없음", not act_sc["seq"]), ("반복글자 없음", not act_sc["rep"]), ("연도·생일 없음", not act_sc["yr"])
            ]
            ch = "".join(
                f'<div class="chip {"na" if v is None else "ok" if v else "no"}">{"–" if v is None else "✓" if v else "✕"} {k}</div>'
                for k, v in chips
            )
            time_sub = "흔한 비번 → 해커가 1순위로 대입" if act_sc["common"] else "글자 조합 + 패턴 보정 시간"

            md(f"""<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                <div class="ct">세부 진단 & 안전 조건 체크</div>
                <span class="grade-pill" style="background:{g_col}">{grade} · {round(s)}점</span>
            </div>
            <div class="cs" style="margin-bottom:8px">빨간색(✕)으로 표시된 항목을 보완할수록 안전도가 빠르게 올라갑니다.</div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-lbl">예상 버팀 시간</div>
                    <div class="stat-val">{human(act_sc['sec'])}</div>
                    <div class="stat-sub">{time_sub}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-lbl">25,000개 중 안전 순위</div>
                    <div class="stat-val">{rank_text(act_sc['pct'])}</div>
                    <div class="stat-sub">상위권일수록 더 안전함</div>
                </div>
            </div>
            {common_html}
            {hibp_html}
            <div style="font-size:13px;font-weight:800;color:#0f172a;margin:12px 0 6px">9가지 안전 비밀번호 조건 {"(실험 반영)" if wi_key != "base" else ""}</div>
            <div class="chk-grid">{ch}</div>""")

    source("t3")

# =========================================================
# TAB 2. 해킹 방법과 공격 속도 비교
# =========================================================
def p_intro():
    d = st.session_state["diag"]
    u_len = d["scenarios"]["base"]["L"]
    u_combo = d["combo"]

    top_c1, top_c2 = st.columns([5.3, 4.7], gap="medium", vertical_alignment="center")
    with top_c1:
        md(f"""<div class="pbanner">
            <div>
                <span class="pstep">STEP 2 · 공격 환경과 해킹 원리</span>
                <div class="ptitle">해커가 공격하는 방식에 따라 뚫리는 속도는 얼마나 다를까?</div>
                <div class="psub">로그인 창에서 직접 찍어볼 때와, 서버가 해킹당해 유출된 파일로 돌릴 때의 속도 차이를 25,500개 데이터로 비교합니다.</div>
                <div style="margin-top:10px"><span class="sync-pill">📌 1번 탭 내 비밀번호 연동 중: {u_len}글자 ({u_combo})</span></div>
            </div>
        </div>""")
    with top_c2:
        with st.container(key="ucard_t2_flt"):
            TS = {"1분": 60, "1시간": 3600, "1일": 86400, "1년": YR, "100년": YR * 100}
            f1, f2 = st.columns(2, gap="medium")
            tk = f1.select_slider("얼마나 버텨야 '안전'하다고 볼까요?", options=list(TS), value="1년", key="i_t")
            T = TS[tk]
            if "i_l" not in st.session_state:
                st.session_state["i_l"] = int(np.clip(u_len, 4, 24))
            Ln = f2.slider("비교해 볼 비밀번호 글자 수", 4, 24, key="i_l")

    pc = [
        (M.crack_time_sec <= T).mean() * 100,
        (P.online_sec <= T).mean() * 100,
        (P.offline_crack_sec <= T).mean() * 100
    ]
    slp = float(np.polyfit(M.length, M.lc, 1)[0])

    summary_cards([
        ("📊", "분석한 실제 비밀번호", f"{len(M) + len(P):,}개", f"구조 분석 {len(M):,}개 + 흔한 비번 {len(P)}개"),
        ("🚪", f"로그인 창 찍기 ({tk} 기준)", f"{pc[1]:.0f}% 뚫림", "초당 10번씩 직접 입력할 때 뚫리는 비율"),
        ("⚡", "서버 파일 유출 시 속도", "1억 배 빠름", "초당 10억 번 대입 (온라인보다 10⁸배 고속)"),
        ("➕", "글자 1개 더 늘릴 때마다", f"약 {10 ** slp:.0f}배 더 버팀", "맞춰야 할 경우의 수가 기하급수로 증가")
    ])

    r1_c1, r1_c2 = st.columns([5.0, 5.0], gap="medium")
    with r1_c1:
        with st.container(key="ucard_t2_wf"):
            lo_l, hi_l = max(4, Ln - 2), Ln + 2
            sub_l = M[M.length.between(lo_l, hi_l)]
            pl = float((sub_l.crack_time_sec <= T).mean() * 100) if len(sub_l) else 0.0
            wf = [pl, pc[1], pc[2]]
            card_head(
                "비밀번호 100개 중 몇 개가 위험할까?",
                f"선택한 안전 기준({tk}) 안에 뚫리는 비밀번호의 비율(빨간 칸)입니다. 서버 파일이 유출되면 위험도가 급증합니다."
            )
            ttl = [f"비슷한 길이 ({lo_l}~{hi_l}자)", "흔한 비번 (로그인 창)", "흔한 비번 (서버 유출 시)"]
            fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=[f"{t}<br><b>{p:.0f}개 뚫림 ({p:.0f}%)</b>" for t, p in zip(ttl, wf)])
            for i, p in enumerate(wf):
                z = np.zeros(100)
                z[:int(round(p))] = 1
                fig.add_trace(go.Heatmap(z=z.reshape(10, 10), colorscale=[[0, "#d1fae5"], [1, "#f87171"]], showscale=False, xgap=3.5, ygap=3.5, hoverinfo="skip"), 1, i + 1)
                fig.update_xaxes(visible=False, row=1, col=i + 1)
                fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i + 1)
            fig.update_annotations(font_size=12.5)
            show(style(fig, 295, t=58, l=6, r=6, b=8), f"i1_{tk}_{Ln}")

    with r1_c2:
        with st.container(key="ucard_t2_bar"):
            def _med_sec(cond, fb_len):
                s_ = M.loc[cond, "crack_time_sec"]
                if len(s_) < 10:
                    s_ = M.loc[M.length == fb_len, "crack_time_sec"]
                if s_.empty:
                    s_ = M.loc[M.length.between(fb_len - 1, fb_len + 1), "crack_time_sec"]
                return float(s_.median())

            ln_med = _med_sec(M.length == Ln, Ln)
            verdict = "안전 기준보다 오래 버팁니다" if ln_med > T else "안전 기준보다 빨리 뚫립니다"
            card_head(
                "특수문자를 넣을까, 글자 수를 늘릴까?",
                f"선택한 {Ln}글자의 보통 버팀 시간(중앙값)은 {human(ln_med)}이며, {verdict}."
            )
            groups = [
                ("8글자 · 특수문자 포함", _med_sec((M.length == 8) & (M.n_special > 0), 8)),
                ("10글자 · 특수문자 없음", _med_sec((M.length == 10) & (M.n_special == 0), 10)),
                ("12글자 · 특수문자 없음", _med_sec((M.length == 12) & (M.n_special == 0), 12)),
                ("16글자 · 특수문자 없음", _med_sec((M.length == 16) & (M.n_special == 0), 16)),
                (f"비교 중인 길이 ({Ln}글자)", ln_med),
            ]
            y_lbls = [g[0] for g in groups]
            meds = [g[1] for g in groups]
            cols = [GREEN if v > T else RED for v in meds]
            l_meds = [float(np.clip(math.log10(max(v, 1e-3)), -3, 18)) for v in meds]
            txt_lbl = [human(v) for v in meds]
            t_pairs = [(0.2, "0.01초"), (3.2, "1초"), (6.756, "1시간"), (10.699, "1년"), (12.699, "100년")]
            if max(l_meds) > 12.0:
                t_pairs.append((18.699, "1억 년"))

            fig = go.Figure(go.Bar(
                y=y_lbls, x=[v + 3.2 for v in l_meds], orientation="h",
                marker=dict(color=cols, line=dict(color="#0f172a", width=[0, 0, 0, 0, 2.8])),
                text=txt_lbl, textposition="outside", cliponaxis=False,
                textfont=dict(size=12.5, weight="bold", color="#0f172a"),
                hovertemplate="%{y}: %{text}<extra></extra>"
            ))
            fig.add_vline(x=math.log10(T) + 3.2, line_dash="dot", line_color="#0f172a", line_width=2, annotation_text=f"기준 ({tk})", annotation_position="top")
            fig.update_layout(bargap=0.36)
            fig.update_xaxes(
                tickvals=[p[0] for p in t_pairs],
                ticktext=[p[1] for p in t_pairs],
                range=[0, max(max(l_meds), math.log10(T)) + 6.2]
            )
            fig.update_yaxes(showgrid=False)
            show(style(fig, 295, t=30, l=10, r=32, b=10), f"i2_{tk}_{Ln}")

    with st.container(key="ucard_t2_atk"):
        card_head(
            "해커들은 실제로 어떤 순서로 비밀번호를 뚫을까?",
            "처음부터 무작정 모든 글자를 다 넣어보지 않습니다. 가장 쉽게 뚫리는 방법부터 1 → 4단계 순서로 공격합니다."
        )
        md("""<div class="atk-grid">
            <div class="atk-step s1">
                <div class="atk-badge"><span class="atk-num">1순위 · 1초 컷</span><span class="atk-tech">사전 공격 (Dictionary)</span></div>
                <div class="atk-ttl">흔한 비밀번호 목록부터 대입</div>
                <div class="atk-txt">사람들이 자주 쓰는 <b>password, qwer1234</b> 같은 유출 빈출 단어장부터 먼저 넣어 계산 시간과 상관없이 1초 안에 엽니다.</div>
            </div>
            <div class="atk-step s2">
                <div class="atk-badge"><span class="atk-num">2순위 · 연쇄 해킹</span><span class="atk-tech">크레덴셜 스터핑</span></div>
                <div class="atk-ttl">털린 옛날 비밀번호 그대로 대입</div>
                <div class="atk-txt">다른 사이트 해킹 사고 때 유출된 아이디·비밀번호(HIBP 기록)를 포털·SNS 로그인 창에 자동으로 똑같이 넣어봅니다.</div>
            </div>
            <div class="atk-step s3">
                <div class="atk-badge"><span class="atk-num">3순위 · 습관 추측</span><span class="atk-tech">규칙 기반 공격 (Rule)</span></div>
                <div class="atk-ttl">생일·연도·키보드 순서 찍기</div>
                <div class="atk-txt">첫 글자만 대문자, 맨 끝에 <b>2024!, 0315, 1234</b>를 붙이는 사람들의 흔한 습관 패턴을 조합해 집중적으로 찌릅니다.</div>
            </div>
            <div class="atk-step s4">
                <div class="atk-badge"><span class="atk-num">4순위 · 최후 수단</span><span class="atk-tech">무차별 대입 (Nᴸ 연산)</span></div>
                <div class="atk-ttl">모든 글자 조합을 끝까지 돌리기</div>
                <div class="atk-txt">위 3가지가 모두 안 통할 때만 가능한 글자를 전부 대입합니다. <b>대시보드의 '예상 버팀 시간'이 바로 이 4단계 기준입니다.</b></div>
            </div>
        </div>""")

    source("t0")

# =========================================================
# TAB 3. 길이와 문자 조합 데이터 검증 (25,000건)
# =========================================================
def p_leak():
    d = st.session_state["diag"]
    u_len = d["scenarios"]["base"]["L"]
    u_combo = d["combo"]

    top_c1, top_c2 = st.columns([5.3, 4.7], gap="medium", vertical_alignment="center")
    with top_c1:
        md(f"""<div class="pbanner">
            <div>
                <span class="pstep">STEP 3 · 25,000건 데이터 검증</span>
                <div class="ptitle">비밀번호를 길고 복잡하게 만들면 실제로 얼마나 오래 버틸까?</div>
                <div class="psub">실제 비밀번호 구조 데이터 25,000건을 통해 글자 수와 문자 조합(영문·숫자·특수문자)이 버팀 시간에 미치는 영향을 검증합니다.</div>
                <div style="margin-top:10px"><span class="sync-pill">📌 내 비밀번호 위치 표시 중: {u_len}글자 · '{u_combo}' 그룹</span></div>
            </div>
        </div>""")
    with top_c2:
        with st.container(key="ucard_t3_flt"):
            f1, f2 = st.columns([1.05, 1.25], gap="medium", vertical_alignment="bottom")
            gs = multi_pick(f1, "보안 강도 등급", ORDER, "a_g")
            lr = f2.slider("살펴볼 글자 수 범위", 4, 63, (4, 40), key="a_l")

    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다. 등급이나 글자 수 범위를 넓혀 보세요.")
        return

    ent_col = "entropy" if "entropy" in f.columns else "shannon"
    summary_cards([
        ("📏", "보통 글자 수 (중앙값)", f"{float(f.length.median()):.0f}글자", f"선택된 {len(f):,}개 표본의 중간 길이"),
        ("🧩", "글자 다양성 점수", f"{float(f[ent_col].median()):.1f} bit", "높을수록 다양한 글자 사용 (엔트로피)"),
        ("⏱️", "보통 버티는 시간", human(f.crack_time_sec.median()), "무차별 대입 기준 중간 버팀 시간"),
        ("⚠️", "1년 안에 뚫리는 비율", f"{(f.crack_time_sec <= YR).mean() * 100:.0f}%", "최소 12글자 이상이어야 안정권 진입")
    ])

    g = f.groupby("length").agg(n=("lc", "size"), med=("crack_time_sec", "median")).query("n>=20").reset_index()
    if f.length.nunique() > 1:
        slope = np.polyfit(f.length, f.lc, 1)[0]
        r2 = float(np.corrcoef(f.length, f.lc)[0, 1] ** 2)
    else:
        slope, r2 = float("nan"), float("nan")

    r1_c1, r1_c2 = st.columns([5.4, 4.6], gap="medium")
    with r1_c1:
        with st.container(key="ucard_t3_line"):
            desc = (
                f"1글자 길어질 때마다 버팀 시간이 평균 약 {10 ** slope:.0f}배씩 폭발적으로 늘어납니다 (추세 설명력 R²={r2:.2f})."
                if np.isfinite(slope) else "글자 수 범위를 넓히면 증가 추세가 나타납니다."
            )
            tbl_a1 = (
                lambda: st.dataframe(
                    g.assign(중간_버팀시간=g.med.map(human))[["length", "n", "중간_버팀시간"]]
                    .rename(columns={"length": "글자 수", "n": "데이터 수"}),
                    hide_index=True
                )
            ) if not g.empty else None
            card_head("글자 수가 늘어날수록 버팀 시간은 어떻게 뛸까?", desc, table_fn=tbl_a1)
            if g.empty:
                st.info("표본이 부족합니다.")
            else:
                y = np.log10(g.med.clip(lower=1e-2))
                fig = go.Figure(go.Scatter(
                    x=g.length, y=y, mode="lines+markers",
                    line=dict(color=BLUE, width=3.2, shape="spline"), marker=dict(size=7.5),
                    customdata=[[human(v), n] for v, n in zip(g.med, g.n)],
                    hovertemplate="길이 %{x}글자<br>보통 버팀 시간: %{customdata[0]}<br>분석 개수: %{customdata[1]}개<extra></extra>"
                ))
                fig.add_hline(y=math.log10(YR), line_dash="dot", line_color=RED, annotation_text="안전 기준 (1년)", annotation_position="top left")
                if lr[0] <= u_len <= lr[1]:
                    fig.add_vline(x=u_len, line_dash="dash", line_color=GREEN, line_width=2.2, annotation_text=f"내 비밀번호 ({u_len}자)", annotation_position="bottom right")
                fig.update_xaxes(title="비밀번호 글자 수 (자)")
                tick_axis(fig, min(y.min(), math.log10(YR)), y.max())
                show(style(fig, 310, t=20, l=10, r=18, b=12), "a1")

    with r1_c2:
        with st.container(key="ucard_t3_box"):
            card_head(
                "무엇을 섞어 써야 더 오래 버틸까?",
                f"숫자만 쓰면 한 칸에 후보가 10개뿐이지만, 영문·특수문자를 섞으면 크게 상승합니다. (내 유형: {u_combo})"
            )
            type_order = ["숫자만", "영문만", "영문+숫자", "특수문자 포함"]
            type_cols = [RED, ORANGE, BLUE, GREEN]
            fig = go.Figure()
            for tp, c_ in zip(type_order, type_cols):
                sub_t = f[f["조합유형"] == tp]
                if not sub_t.empty:
                    lbl = f"★ {tp} (내 유형)" if tp == u_combo else tp
                    fig.add_trace(go.Box(y=sub_t["lc"].clip(lower=-2), name=lbl, marker_color=c_, boxpoints=False))
            tick_axis(fig, max(f["lc"].min(), -2), f["lc"].max(), "y")
            fig.update_layout(showlegend=False)
            show(style(fig, 310, t=20, l=10, r=16, b=12), "a3")

    with st.container(key="ucard_t3_surv"):
        stp = [
            ("처음 전체", 0), ("1분 후 생존", 60), ("1시간 후 생존", 3600),
            ("하루(1일) 후 생존", 86400), ("1년 후 생존 (안전)", YR), ("100년 후 생존", YR * 100)
        ]
        vals = [100.0] + [(f.crack_time_sec > t).mean() * 100 for _, t in stp[1:]]
        card_head(
            "시간이 지날수록 비밀번호 100개 중 몇 개가 살아남을까?",
            f"선택한 데이터 중 하루를 버티는 비밀번호는 {vals[3]:.0f}개, 1년 이상 안전하게 살아남는 비밀번호는 {vals[4]:.0f}개입니다."
        )
        fig_f = go.Figure(go.Bar(
            x=[a for a, _ in stp], y=vals,
            text=[f"<b>{v:.0f}% 생존</b>" for v in vals], textposition="outside", cliponaxis=False,
            marker=dict(color=["#64748b", "#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#059669"], cornerradius=10),
            hovertemplate="%{x}: %{y:.1f}% 살아남음<extra></extra>"
        ))
        fig_f.update_yaxes(title="살아남은 비율 (%)", range=[0, 118])
        fig_f.update_xaxes(showgrid=False)
        show(style(fig_f, 235, t=24, l=10, r=16, b=10), "a2")

    source("t1")

# =========================================================
# TAB 4. 흔한 비밀번호 상위 500건 검증
# =========================================================
def p_common():
    top_c1, top_c2 = st.columns([5.3, 4.7], gap="medium", vertical_alignment="center")
    with top_c1:
        md("""<div class="pbanner">
            <div>
                <span class="pstep">STEP 4 · 흔한 비밀번호 Top 500 검증</span>
                <div class="ptitle">'7년 버틴다'고 적힌 비밀번호가 왜 실제로는 1초 만에 털릴까?</div>
                <div class="psub">전 세계 상위 500개 흔한 비밀번호를 통해 글자 수만 믿다 당하는 '보안 착시'와 유출·사전 공격의 위험성을 확인합니다.</div>
            </div>
        </div>""")
    with top_c2:
        with st.container(key="ucard_t4_flt"):
            f1, f2 = st.columns([1.15, 1.15], gap="medium", vertical_alignment="bottom")
            n = f1.slider("인기 순위 몇 위까지 볼까요?", 10, 500, 40, 10, key="b_n")
            allc = sorted(P.cat_ko.unique())
            cats = multi_pick(f2, "비밀번호 단어 유형", allc, "b_c")

    q = P[(P["rank"] <= n) & P.cat_ko.isin(cats)]
    if q.empty:
        st.info("조건에 맞는 비밀번호가 없습니다. 단어 유형을 하나 이상 선택해 주세요.")
        return

    topc = q.cat_ko.value_counts()
    summary_cards([
        ("🚪", "로그인 창 찍기 (보통 시간)", human(q.online_sec.median()), "초당 10번씩 순서대로 입력할 때"),
        ("⚡", "서버 유출 시 (보통 시간)", human(q.offline_crack_sec.median()), "초당 10억 번 고속 대입할 때"),
        ("🕒", "로그인 창 · 하루 내 뚫림", f"{(q.online_sec <= 86400).mean() * 100:.0f}%", f"선택한 {len(q)}개 중 하루 안에 열리는 비율"),
        ("🚨", "서버 유출 · 1초 내 뚫림", f"{(q.offline_crack_sec <= 1).mean() * 100:.0f}%", "흔한 목록 대입(사전 공격) 시에는 100% 즉시 해독")
    ])

    r1_c1, r1_c2 = st.columns([5.3, 4.7], gap="medium")
    with r1_c1:
        with st.container(key="ucard_t4_tile"):
            tbl_b1 = lambda: st.dataframe(
                q.assign(로그인창_대입=q.online_sec.map(human), 서버유출_대입=q.offline_crack_sec.map(human))[
                    ["rank", "password", "cat_ko", "로그인창_대입", "서버유출_대입"]
                ].rename(columns={"rank": "순위", "password": "비밀번호", "cat_ko": "단어 유형"}),
                hide_index=True
            )
            card_head(
                f"전 세계 흔한 비밀번호 상위 {len(q)}개와 '보안 착시'",
                "초록색 타일(예: password · 7년)은 로그인 창 무차별 대입 시간일 뿐, 실제로는 해커 사전 1순위라 1초 안에 뚫립니다.",
                table_fn=tbl_b1
            )
            tile_grid(q)

    with r1_c2:
        with st.container(key="ucard_t4_area"):
            a_, b_ = (q.online_sec <= 86400).mean() * 100, (q.offline_crack_sec <= 86400).mean() * 100
            card_head(
                "로그인 창 입력 vs 서버 유출 시 뚫리는 속도 비교",
                f"하루 안에 로그인 창에서는 {a_:.0f}%가 뚫리지만, 서버 유출 파일로는 {b_:.0f}%가 뚫립니다 (속도 1억 배 차이)."
            )
            fig = go.Figure()
            for nm, col, rgba, v in [
                ("로그인 창 직접 입력 (초당 10번)", BLUE, "rgba(59,130,246,.16)", q.lon),
                ("서버 유출 파일 대입 (초당 10억 번)", RED, "rgba(239,68,68,.14)", q.lof)
            ]:
                xs = np.sort(v.values)
                ys = np.arange(1, len(xs) + 1) / len(xs) * 100
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, name=nm, mode="lines",
                    line=dict(color=col, width=3, shape="hv"), fill="tozeroy", fillcolor=rgba,
                    hovertemplate="이 시간 안에 %{y:.0f}% 뚫림<extra>" + nm + "</extra>"
                ))
            tick_axis(fig, min(q.lon.min(), q.lof.min()), max(q.lon.max(), q.lof.max()), "x")
            fig.update_yaxes(title="뚫리는 비밀번호 비율 (%)", range=[0, 104])
            show(style(fig, 315, t=28, l=10, r=18, b=12), "b3")

    with st.container(key="ucard_t4_cat"):
        cat_df = (
            q.groupby("cat_ko")
            .agg(cnt=("password", "size"), ex=("password", lambda s: ", ".join(s.astype(str).head(3))))
            .reset_index()
            .sort_values("cnt", ascending=True)
        )
        card_head(
            "사람들은 흔한 비밀번호에 어떤 단어를 가장 많이 쓸까?",
            f"가장 많은 유형은 '{topc.index[0]}'({topc.iloc[0]}개, {topc.iloc[0] / len(q):.0%})입니다. 막대에 마우스를 올리면 실제 자주 쓰인 예시 단어가 보입니다."
        )
        fig_c = go.Figure(go.Bar(
            y=cat_df["cat_ko"], x=cat_df["cnt"], orientation="h",
            marker=dict(color=BLUE, cornerradius=8),
            text=[f"{c}개 ({c / len(q):.0%})  ·  예: {e}" for c, e in zip(cat_df["cnt"], cat_df["ex"])],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=12.5, color="#1e293b"),
            hovertemplate="<b>%{y}</b>: %{x}개<extra></extra>"
        ))
        fig_c.update_xaxes(title="포함된 비밀번호 개수", range=[0, max(cat_df["cnt"]) * 1.65])
        fig_c.update_yaxes(showgrid=False)
        show(style(fig_c, max(220, len(cat_df) * 32 + 45), t=14, l=12, r=36, b=12), "b2")

    source("t2")

if page == 0:
    p_lab()
elif page == 1:
    p_intro()
elif page == 2:
    p_leak()
else:
    p_common()
