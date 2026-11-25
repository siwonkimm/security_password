import hashlib, html as _html, inspect, math, re, urllib.request
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
YR = 31557600
CAND = [1e-3, 1, 60, 3600, 86400, YR, YR * 1e2, YR * 1e4, YR * 1e6, YR * 1e9, YR * 1e12, YR * 1e18, YR * 1e24, YR * 1e36]

SRC = {
    "t1": (
        "Password Metrics Dataset (Kaggle) — 고유 25,000건 구조 피처",
        "원본 100,000행 중 중복을 제거한 고유 25,000개(개인정보 보호를 위해 원문은 제거하고 길이·문자 구성·패턴 등 20개 수치 피처만 저장). "
        "여기서 '버팀 시간(초)'은 문자 종류 크기(N)와 길이(L)에 따른 모든 경우의 수(N^L)를 무차별 대입(Brute-force)할 때 걸리는 이론적 연산 시간입니다(범위 10^-7~10^104초, 로그 눈금·중앙값 집계). "
        "[한계] 등급별 약 5,000개씩 추출된 균등 표본이므로 실제 이용자 비율과 다르며, 원본 수식에는 키보드 연속·연도 패턴 감점이 반영되어 있지 않습니다.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset")]
    ),
    "t2": (
        "Password Strength and Vulnerability Dataset (Kaggle / Information is Beautiful) — 상위 500건",
        "전 세계에서 가장 흔한 비밀번호 상위 500개 공개 자료입니다. "
        "① 로그인 창 대입(online_sec)은 웹·앱 로그인 화면 직접 입력 공격(서버 지연으로 초당 약 10회 대입 가정), "
        "② 유출 파일 대입(offline_crack_sec)은 외부로 유출된 해시 파일을 고성능 장비로 대입하는 공격(초당 약 10억 회 가정, 온라인보다 정확히 10^8배 빠름)을 초 단위로 환산(1개월=30.4일, 1년=365.25일)한 값입니다. "
        "[한계] 표기된 시간은 사전 없이 알파벳을 처음부터 대입하는 '순수 무차별 대입' 기준이므로, 실제 해킹 1순위인 '사전 대입 공격(Dictionary Attack)' 적용 시 500개 모두 표기 시간과 무관하게 1초 이내 즉시 뚫립니다.",
        [("원본 데이터 보기 (Kaggle)", "https://www.kaggle.com/datasets/utkarshx27/passwords")]
    ),
    "t3": (
        "자체 학습 예측 모델(Random Forest) · 조건 변경 실험(What-If) · HIBP 실시간 유출 조회(k-익명성)",
        "Kaggle 고유 25,000건의 20개 구조적 피처로 학습한 Random Forest 회귀 모델, 상위 500개 흔한 비밀번호 목록, 그리고 HIBP(Have I Been Pwned) Pwned Passwords API를 결합해 진단합니다. "
        "직접 입력한 비밀번호 원문은 세션에 저장하거나 외부로 전송하지 않으며, HIBP 조회 시에도 SHA-1 해시의 앞 5자리만 전송하는 k-익명성(k-Anonymity) 방식으로 실제 유출 횟수를 안전하게 확인합니다. "
        "유출 횟수는 HIBP 응답값 그대로 표시하며, 이 값으로 버팀 시간을 계산하지 않습니다. 조회에 실패하면 '확인 불가'로 표시하고 안전으로 간주하지 않습니다.",
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

def md(s, pad=True):
    st.markdown("".join(l.strip() for l in s.splitlines()) + ('<div style="height:14px"></div>' if pad else ""), unsafe_allow_html=True)

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
            with st.popover("📋 표", **_POP_W):
                table_fn()
        md(f'<div class="sec" style="min-height:46px;padding-top:0"><div class="k" style="margin-top:0">{take}</div></div>')

def source(key):
    n, t, links = SRC[key]
    with st.expander("데이터 출처 · 시간(초) 계산 기준 · 표본 한계"):
        link_md = " &nbsp;|&nbsp; ".join(f"🔗 [{label}]({url})" for label, url in links)
        st.markdown(f"**{n}**\n\n{t}\n\n{link_md}")

def ins(text, color=BLUE, label="핵심 요약"):
    md(f'<div class="ins" style="border-left-color:{color}"><b>{label}</b>{text}</div>')

def tile_grid(qq):
    lo, hi = 0.0, 7.5
    A, B, Cc = (255, 154, 143), (255, 228, 154), (143, 220, 176)
    def col(v):
        vc = float(np.clip(v, lo, hi))
        t = (vc - lo) / (hi - lo)
        p, q_, u = (A, B, t * 2) if t < .5 else (B, Cc, (t - .5) * 2)
        return "rgb(%d,%d,%d)" % tuple(int(p[k] + (q_[k] - p[k]) * u) for k in range(3))
    cells = "".join(
        f'<div class="tl" style="background:{col(v)};animation-delay:{min(i,20)*30}ms" '
        f'title="{_html.escape(str(w))} · 로그인 창 대입 {human(s_)} / 유출 파일 대입 {human(of_)} (사전 공격 시 즉시)">'
        f'<span class="rk">{int(r)}위</span><b>{_html.escape(str(w))}</b><span class="tm">{human(s_)}</span></div>'
        for i, (r, w, v, s_, of_) in enumerate(zip(qq["rank"], qq.password, qq.lon, qq.online_sec, qq.offline_crack_sec))
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
        f'<div class="v" style="font-size:{25 if isinstance(v,(int,float)) else 18}px">'
        f'<span class="n" data-v="{v if isinstance(v,(int,float)) else ""}" data-d="{d}">'
        f'{v if not isinstance(v,(int,float)) else f"{v:,.{d}f}"}</span><span class="u">{u}</span></div></div>'
        for i, (l, v, d, u) in enumerate(items)
    )
    h = f"""<style>
    body{{margin:0;padding:4px 2px 10px;font-family:{FONT}}}.g{{display:grid;grid-template-columns:repeat({len(items)},minmax(0,1fr));gap:12px}}
    .c{{background:#fff;border-radius:18px;padding:14px 16px 16px;border-left:5px solid #4cc38a;box-shadow:0 2px 12px rgba(0,0,0,.06);animation:u .5s ease both;min-width:0}}
    .l{{color:#6e6e73;font-size:12.3px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .v{{margin-top:6px;color:#1d1d1f;font-weight:700;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.25}}.u{{font-size:13.5px;color:#6e6e73;margin-left:3px}}
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
    p["lon"] = np.log10(p.online_sec.clip(lower=1e-6))
    p["lof"] = np.log10(p.offline_crack_sec.clip(lower=1e-6))
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
html,body,[class*="css"],.stApp{font-family:Pretendard,-apple-system,'Apple SD Gothic Neo',sans-serif!important;overflow-x:hidden}
.stApp{background:#e9edf2}#MainMenu,footer{visibility:hidden}[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}
.block-container{max-width:1520px;width:100%;box-sizing:border-box;padding:.9rem 1.4rem 1.3rem;position:relative;z-index:1;animation:up .6s both}[data-testid="stVerticalBlock"]{gap:.5rem}
@keyframes up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes wipe{from{opacity:0;clip-path:inset(0 100% 0 0 round 22px)}to{opacity:1;clip-path:inset(0 0 0 0 round 22px)}}

/* 상단 헤더 */
.st-key-hdr{background:linear-gradient(100deg,#171f3d,#0f1730);border-radius:22px;padding:11px 20px;border:1px solid #25305a;box-shadow:0 6px 24px rgba(15,23,48,.25)}
.st-key-hdr [data-testid="stHorizontalBlock"]{align-items:center!important;min-height:46px}
.st-key-hdr [data-testid="stColumn"]{display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stElementContainer"],.st-key-hdr [data-testid="stMarkdownContainer"],.st-key-hdr [data-testid="stButton"],.st-key-hdr [data-testid="stButtonGroup"]{margin:0!important;padding:0!important;display:flex!important;align-items:center!important}
.st-key-hdr [data-testid="stMarkdownContainer"] p{margin:0!important;padding:0!important;line-height:46px!important}
.st-key-hdr *{color:#e6e9f2}
.ht{font-size:19px;font-weight:800;letter-spacing:-.02em;white-space:nowrap;display:inline-flex;align-items:center;gap:8px;height:46px;line-height:46px}
.st-key-hdr [data-testid="stButtonGroup"]{flex-wrap:nowrap;gap:8px;height:46px}
.st-key-hdr [data-testid="stButtonGroup"] button{height:44px!important;min-height:44px!important;padding:0 16px!important;font-size:14.5px;font-weight:700;border-radius:13px;background:#1a2547;border:1px solid #2a3868;white-space:nowrap;transition:all .25s;display:inline-flex;align-items:center;justify-content:center}
.st-key-hdr [data-testid="stButtonGroup"] button:hover{transform:translateY(-1px);border-color:#5b8def}
.st-key-hdr [data-testid="stButtonGroup"] button[aria-checked="true"],.st-key-hdr [data-testid="stButtonGroup"] button[aria-pressed="true"],.st-key-hdr button[data-testid="stBaseButton-segmented_controlActive"]{background:#5b8def!important;border-color:#8fb4ff;box-shadow:0 0 18px rgba(91,141,239,.55)}
.st-key-hdr [role="radiogroup"]{flex-direction:row;gap:8px;flex-wrap:nowrap;align-items:center}.st-key-hdr [role="radiogroup"] label{padding:8px 14px;border-radius:13px;background:#1a2547;border:1px solid #2a3868}
.st-key-hdr button[kind="secondary"]{background:#1a2547;border:1px solid #2a3868;border-radius:13px;height:44px!important;min-height:44px!important;padding:0 14px!important}

.sec{padding:3px 2px 3px;min-height:66px}.sec .t{font-size:16px;font-weight:800;letter-spacing:-.02em;line-height:1.35}.sec .k{font-size:12.5px;color:#5b6472;margin-top:4px;line-height:1.5;word-break:keep-all}
.bd{background:#5b8def;color:#fff;font-size:11px;font-weight:700;border-radius:6px;padding:2px 8px;margin-right:8px;vertical-align:2px}
div[data-testid="stPlotlyChart"]{background:#fff;border-radius:22px;overflow:hidden;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:wipe .9s cubic-bezier(.2,.8,.2,1) both;transition:box-shadow .3s}
div[data-testid="stPlotlyChart"]:hover{box-shadow:0 14px 34px rgba(60,90,140,.16)}
.ins{background:#fff;border-left:5px solid #5b8def;border-radius:14px;padding:12px 18px;font-size:13.8px;line-height:1.62;box-shadow:0 2px 12px rgba(0,0,0,.05);animation:up .6s both;word-break:keep-all}.ins>b:first-child{margin-right:8px}

/* 두 가지 공격 기준 안내 카드 */
.meta-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.meta-card{background:#fff;border-radius:16px;padding:11px 16px;border:1px solid #e2e8f2;box-shadow:0 2px 10px rgba(0,0,0,.04);display:flex;align-items:center;gap:12px;word-break:keep-all}
.meta-ic{font-size:24px;flex:none;background:#f2f6ff;width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center}
.meta-card.key .meta-ic{background:#fff2f0}
.meta-ttl{font-size:13.5px;font-weight:800;color:#1d1d1f;margin-bottom:2px}
.meta-dsc{font-size:12.2px;color:#5b6472;line-height:1.45}

/* 진단 배너 · What-If 바 · 결과 카드 */
.diag-target{background:#fff;border-radius:16px;padding:10px 18px;box-shadow:0 2px 12px rgba(0,0,0,.05);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;border:1px solid #e2e8f2;animation:up .4s both}
.diag-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:800;padding:4px 10px;border-radius:8px;background:#eef3ff;color:#2b5fc7}
.diag-badge.user{background:#f2f4f8;color:#4b5563}
.diag-code{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:14px;font-weight:800;background:#f5f7fa;color:#1d1d1f;padding:3px 11px;border-radius:8px;border:1px solid #dce2ec;letter-spacing:.03em}
.wi-banner{background:#f5f8ff;border:1px solid #d6e4ff;border-radius:14px;padding:10px 16px;font-size:13.2px;line-height:1.55;color:#1d1d1f;word-break:keep-all}
.wi-banner.warn{background:#fff6f5;border-color:#ffd5cf}
.res-card{background:#fff;border-radius:22px;padding:18px 20px;box-shadow:0 2px 14px rgba(0,0,0,.05);animation:up .6s both;min-height:462px;display:flex;flex-direction:column;justify-content:space-between;box-sizing:border-box}
.res-hdr{font-size:16px;font-weight:800;color:#1d1d1f;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.grade-pill{font-size:12.5px;font-weight:800;padding:4px 11px;border-radius:99px;color:#fff!important}
.stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-bottom:10px}
.stat-box{background:#f5f7fa;border-radius:15px;padding:11px 13px;border:1px solid #e8ecf2}
.stat-lbl{font-size:11.8px;font-weight:600;color:#6e6e73;margin-bottom:3px}
.stat-val{font-size:19.5px;font-weight:800;color:#1d1d1f;letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-sub{font-size:11.5px;color:#5b6472;font-weight:600;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hibp-bar{border-radius:12px;padding:8px 13px;margin-bottom:7px;font-size:12.4px;font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:8px}
.hibp-bad{background:#ffeceb;color:#c92a20;border:1px solid #ffd1ce}
.hibp-ok{background:#e6f7ed;color:#157332;border:1px solid #c6ebd4}
.hibp-na{background:#f5f7fa;color:#5b6472;border:1px solid #e2e6ee}
.chk-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:6px}
.chip{display:flex;align-items:center;justify-content:center;border-radius:10px;padding:8px 5px;font-size:12px;font-weight:700;text-align:center;white-space:nowrap}.ok{background:#e6f7ed;color:#157332;border:1px solid #c6ebd4}.no{background:#ffeceb;color:#c92a20;border:1px solid #ffd1ce}.na{background:#f2f4f8;color:#5b6472;border:1px solid #e2e6ee}
.tip-item{background:#f8f9fc;border-radius:12px;padding:11px 13px;margin-bottom:9px;font-size:12.9px;line-height:1.52;color:#1d1d1f;border-left:4px solid #5b8def;word-break:keep-all}
.tip-item.warn{border-left-color:#ff3b30;background:#fff8f7}
.tip-item.good{border-left-color:#4cc38a;background:#f4fbf7}
.tip-tag{display:inline-block;font-size:11px;font-weight:800;padding:1px 7px;border-radius:6px;margin-right:6px;background:#e5edff;color:#2b5fc7}
.tip-item.warn .tip-tag{background:#ffe3e0;color:#c92a20}
.tip-item.good .tip-tag{background:#dcf5e6;color:#157332}

span[data-baseweb="tag"]{background:#5b8def!important;border-radius:8px}[role="slider"]{background:#5b8def!important}[data-testid="stExpander"]{border-radius:14px;background:#fff}
[data-testid="stPopover"] button{border-radius:99px!important;padding:2px 10px!important;min-height:30px!important;font-size:12px!important;font-weight:700!important;background:#fff!important;border:1px solid #d5dbe5!important;box-shadow:0 1px 4px rgba(0,0,0,.04)!important}
[data-testid="stPopover"] button:hover{border-color:#5b8def!important;color:#5b8def!important}
.spl{position:fixed;inset:0;z-index:99999;background:#000;display:flex;align-items:center;justify-content:center;pointer-events:none;animation:so 2.4s ease forwards}
.spl div{color:#fff;font-size:clamp(30px,5.8vw,62px);font-weight:800;letter-spacing:-.03em;animation:si 1.3s cubic-bezier(.2,.8,.2,1) both}
@keyframes si{from{opacity:0;transform:scale(.92);filter:blur(10px)}to{opacity:1;transform:none;filter:none}}@keyframes so{0%,72%{opacity:1}100%{opacity:0;visibility:hidden}}
.tg{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;background:#fff;border-radius:22px;padding:14px;box-shadow:0 2px 14px rgba(0,0,0,.05);height:310px;overflow-y:auto;box-sizing:border-box;align-content:start}.tg::-webkit-scrollbar{width:6px}.tg::-webkit-scrollbar-thumb{background:#cfd6e4;border-radius:6px}.tl{border-radius:14px;padding:10px 6px;text-align:center;min-width:0;display:flex;flex-direction:column;gap:3px;justify-content:center;animation:up .4s both;transition:transform .25s}.tl:hover{transform:translateY(-3px) scale(1.03)}.tl b{font-size:14px;color:#1d1d1f;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rk{font-size:11px;color:rgba(0,0,0,.55);font-weight:600}.tm{font-size:12px;font-weight:700;color:rgba(0,0,0,.75);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.igs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.ig{display:flex;gap:14px;align-items:center;border-radius:22px;padding:16px 18px;color:#fff;animation:up .6s both;min-width:0;box-shadow:0 8px 22px rgba(60,90,140,.18);transition:transform .25s}.ig:hover{transform:translateY(-4px)}.ig:nth-child(1){background:linear-gradient(135deg,#4f7fe6,#7aa7ff)}.ig:nth-child(2){background:linear-gradient(135deg,#f0554a,#ff8f7a)}.ig:nth-child(3){background:linear-gradient(135deg,#e8860a,#ffb347)}.ig:nth-child(4){background:linear-gradient(135deg,#2ea36f,#6fd6a3)}.ic{font-size:36px;flex:none;filter:drop-shadow(0 4px 6px rgba(0,0,0,.18))}.ib{font-size:27px;font-weight:800;letter-spacing:-.02em;line-height:1.1;white-space:nowrap}.ib small{font-size:13px;font-weight:700;margin-left:3px;opacity:.92}.icp{font-size:12px;line-height:1.45;margin-top:5px;opacity:.96;word-break:keep-all}

html{color-scheme:light}[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] strong,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *,[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"],[data-testid="stCheckbox"] label *,[data-testid="stToggle"] label *{color:#1d1d1f!important}.sec .t,.ins,.ins b{color:#1d1d1f}.sec .k{color:#5b6472}.ins{color:#1d1d1f}div.st-key-hdr,div.st-key-hdr *,div.st-key-hdr p,div.st-key-hdr button *{color:#e6e9f2!important}div.ig,div.ig *{color:#fff!important}

.st-key-chk_panel,.st-key-wi_panel{background:#fff;border-radius:20px;padding:12px 16px;box-shadow:0 2px 14px rgba(0,0,0,.05)}
.st-key-chk_panel [data-testid="stHorizontalBlock"],.st-key-wi_panel [data-testid="stHorizontalBlock"]{align-items:center!important}
.st-key-d_ex0 button,.st-key-d_ex1 button,.st-key-d_ex2 button,.st-key-d_ex3 button{width:100%;min-height:42px;padding:0 8px!important;border-radius:13px;border:0;box-shadow:0 4px 12px rgba(60,90,140,.16);transition:transform .2s,box-shadow .2s;white-space:nowrap}
.st-key-d_ex0 button{background:linear-gradient(135deg,#f0554a,#ff8f7a)}
.st-key-d_ex1 button{background:linear-gradient(135deg,#e8860a,#ffb347)}
.st-key-d_ex2 button{background:linear-gradient(135deg,#4f7fe6,#7aa7ff)}
.st-key-d_ex3 button{background:linear-gradient(135deg,#2ea36f,#6fd6a3)}
.st-key-d_ex0 button:hover,.st-key-d_ex1 button:hover,.st-key-d_ex2 button:hover,.st-key-d_ex3 button:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(60,90,140,.24)}
.st-key-d_ex0 button *,.st-key-d_ex1 button *,.st-key-d_ex2 button *,.st-key-d_ex3 button *{color:#fff!important;font-size:12.3px;font-weight:700}

.st-key-wi_panel button{min-height:40px!important;border-radius:12px!important;font-weight:700!important;font-size:12.8px!important;transition:all .2s}
.st-key-wi_panel button[kind="primary"]{background:#5b8def!important;border:1px solid #4a7de0!important;box-shadow:0 4px 14px rgba(91,141,239,.35)!important}
.st-key-wi_panel button[kind="primary"],.st-key-wi_panel button[kind="primary"] *,.st-key-wi_panel button[kind="primary"] p{color:#fff!important;font-weight:800!important}
.st-key-wi_panel button[kind="secondary"]{background:#f4f6fb!important;border:1px solid #dce3ef!important}
.st-key-wi_panel button[kind="secondary"] *{color:#1d1d1f!important;font-weight:700!important}

[data-testid="stForm"] button{background:#5b8def;border:0;border-radius:13px;min-height:42px;width:100%}[data-testid="stForm"] button *{color:#fff!important;font-weight:700;font-size:13px}[data-testid="stForm"] input{border-radius:13px;min-height:42px}[data-testid="InputInstructions"]{display:none!important}</style>""", False)

if not st.session_state.get("splash"):
    st.session_state["splash"] = True
    md('<div class="spl"><div>내 비밀번호, 얼마나 버틸까?</div></div>', False)

NAMES = [
    "🔍 1. 내 비밀번호 실험",
    "⚡ 2. 공격 방식과 속도",
    "📏 3. 길이와 문자 조합",
    "⚠️ 4. 흔한 비밀번호"
]
if "nav" not in st.session_state: st.session_state["nav"] = NAMES[0]
def _keep():
    if st.session_state.get("nav") is None: st.session_state["nav"] = st.session_state.get("last_nav", NAMES[0])

def _go_nav(target_name: str):
    st.session_state["nav"] = target_name
    st.session_state["last_nav"] = target_name

with st.container(key="hdr"):
    hc = st.columns([2.35, 8.45, 1.2], vertical_alignment="center")
    hc[0].markdown('<div class="ht">🔐 비밀번호 보안 분석</div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        hc[1].segmented_control("메뉴", NAMES, key="nav", label_visibility="collapsed", on_change=_keep)
    else:
        hc[1].radio("메뉴", NAMES, horizontal=True, label_visibility="collapsed", key="nav")
    hc[2].button("▶ 인트로", on_click=lambda: st.session_state.update(splash=False), **_BTN_W)

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

def filters(n=3): return st.columns([2.2, 2.2, 2.8][:n], vertical_alignment="center")

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
    return dict(
        L=L, adj_lc=adj_lc, raw_sec=raw_sec, pct=pct, score=score, common=common,
        seq=seq, rep=rep, yr=yr,
        up=feats["n_upper"] > 0, dg=feats["n_digit"] > 0, sp=(feats["n_special"] + feats["n_non"]) > 0
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

    def _pack(st_dict, key, label, short, preview_str, apply_leak=False):
        sec = min(st_dict["raw_sec"], 1.0) if st_dict["common"] else st_dict["raw_sec"]
        sc = st_dict["score"]
        if st_dict["common"]:
            sc = min(sc, 5.0)
        elif apply_leak and leaked:
            sc = min(sc, 10.0 if hibp_cnt < 100 else 5.0)
        return dict(
            key=key, label=label, short=short,
            preview=preview_str if is_ex else None,
            L=st_dict["L"], sec=sec, raw_sec=st_dict["raw_sec"],
            pct=st_dict["pct"], score=sc, common=st_dict["common"],
            seq=st_dict["seq"], rep=st_dict["rep"], yr=st_dict["yr"],
            up=st_dict["up"], dg=st_dict["dg"], sp=st_dict["sp"]
        )

    scenarios = {
        "base": _pack(base, "base", "현재 입력값", "현재 입력값", pw, apply_leak=True),
        "sp1": _pack(s_sp, "sp1", "특수문자 1개 추가", "+특수문자 1개", pw_sp, apply_leak=False),
        "len4": _pack(s_l4, "len4", "길이 4자 추가", "+길이 4자", pw_l4, apply_leak=False),
        "both": _pack(s_both, "both", "특수문자 + 4자 추가", "+특수문자·4자", pw_both, apply_leak=False)
    }

    st.session_state["diag"] = dict(
        hibp=hibp_cnt, is_example=is_ex, example_label=example_label,
        base_common=base["common"], scenarios=scenarios
    )
    st.session_state["wi_mode"] = "base"

if "diag" not in st.session_state:
    diagnose("latte79", example_label="짧은 일상형")
if "wi_mode" not in st.session_state:
    st.session_state["wi_mode"] = "base"

def p_lab():
    head(
        "질문 1. 내 비밀번호에 특수문자나 글자 수를 더하면 버팀 시간이 얼마나 달라질까?",
        "입력한 비밀번호는 저장되지 않으며, 유출 조회 시에도 SHA-1 해시 앞 5자리만 전송(k-익명성)해 안전하게 진단합니다."
    )

    md("""<div class="meta-grid">
        <div class="meta-card">
            <div class="meta-ic">🔢</div>
            <div>
                <div class="meta-ttl">진단 기준 1 · 모든 조합 대입 (예상 버팀 시간)</div>
                <div class="meta-dsc">가능한 글자 조합을 처음부터 하나씩 맞춰볼 때 걸리는 시간입니다. 글자 수와 문자 종류를 늘릴수록 기하급수적으로 길어집니다.</div>
            </div>
        </div>
        <div class="meta-card key">
            <div class="meta-ic">⚠️</div>
            <div>
                <div class="meta-ttl">진단 기준 2 · 유출·흔한 목록 대조 (사전 공격)</div>
                <div class="meta-dsc">상위 500개 흔한 비밀번호나 실제 유출 목록(HIBP)에 이미 포함된 비밀번호는 계산상 버팀 시간과 무관하게 <b>1초 이내</b>에 뚫립니다.</div>
            </div>
        </div>
    </div>""")

    with st.container(key="chk_panel"):
        md('<div style="font-size:13px;font-weight:700;color:#1d1d1f;margin-bottom:7px">직접 입력하거나, 오른쪽 4가지 대표 유형을 눌러 바로 비교해 보세요</div>', pad=False)
        top_row = st.columns([4.0, 1.5, 1.5, 1.5, 1.5], vertical_alignment="center")
        with top_row[0]:
            with st.form("f", clear_on_submit=True, border=False):
                fc = st.columns([3.6, 1.4], vertical_alignment="center")
                pw = fc[0].text_input("비밀번호 입력", type="password", placeholder="비밀번호 입력 (화면에 표시·저장 안 됨)", label_visibility="collapsed")
                if fc[1].form_submit_button("진단하기") and pw:
                    diagnose(pw, example_label=None)
        examples = [
            ("🔓 흔한 단어형", "흔한 단어형", "password123"),
            ("☕ 짧은 일상형", "짧은 일상형", "latte79"),
            ("🧩 연도·규칙형", "연도·규칙형", "Summer2024!"),
            ("🎲 무작위 안전형", "무작위 안전형", "vK9#mQ2$zL7@pW4xR8")
        ]
        for i, (col, (btn_lab, ex_name, ex_val)) in enumerate(zip(top_row[1:], examples)):
            if col.button(btn_lab, key=f"d_ex{i}", **_BTN_W):
                diagnose(ex_val, example_label=ex_name)

    d = st.session_state["diag"]
    scs = d["scenarios"]
    wi_key = st.session_state.get("wi_mode", "base")
    if wi_key not in scs:
        wi_key = "base"
    base_sc = scs["base"]
    act_sc = scs[wi_key]
    hibp_cnt = d.get("hibp")
    leaked = hibp_cnt is not None and hibp_cnt > 0

    with st.container(key="wi_panel"):
        wc = st.columns([2.2, 1.9, 1.95, 1.95, 2.0], vertical_alignment="center")
        wc[0].markdown('<div style="font-size:13.2px;font-weight:800;color:#1d1d1f">🧪 조건 변경 실험 (What-If)</div>', unsafe_allow_html=True)
        wi_opts = [
            ("base", "현재 입력값"),
            ("sp1", "+ 특수문자 1개"),
            ("len4", "+ 길이 4자"),
            ("both", "+ 특수문자 & 4자")
        ]
        for idx, (k_, lbl_) in enumerate(wi_opts):
            if wc[idx + 1].button(lbl_, key=f"wi_b{idx}", type="primary" if wi_key == k_ else "secondary", **_BTN_W):
                st.session_state["wi_mode"] = k_
                st.rerun()

    if d.get("is_example"):
        code_txt = f"{base_sc['preview']} → {act_sc['preview']}" if wi_key != "base" else base_sc["preview"]
        md(
            f'<div class="diag-target">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span class="diag-badge">예시 선택 ({_html.escape(d["example_label"])})</span>'
            f'<span class="diag-code">{_html.escape(str(code_txt))}</span>'
            f'</div>'
            f'<span style="font-size:12.5px;color:#5b6472;font-weight:700">길이 {act_sc["L"]}자</span>'
            f'</div>',
            pad=False
        )
    else:
        mode_note = f"직접 입력 진단 ({act_sc['label']} 적용)" if wi_key != "base" else "직접 입력 진단"
        md(
            f'<div class="diag-target">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span class="diag-badge user">🔒 {mode_note}</span>'
            f'<span style="font-size:12.5px;font-weight:600;color:#5b6472">입력한 비밀번호는 보안상 화면에 표시하거나 저장하지 않습니다.</span>'
            f'</div>'
            f'<span style="font-size:12.5px;color:#5b6472;font-weight:700">길이 {act_sc["L"]}자</span>'
            f'</div>',
            pad=False
        )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    mult_sp = max(1.0, scs["sp1"]["sec"] / max(base_sc["sec"], 1e-6))
    mult_act = max(1.0, act_sc["sec"] / max(base_sc["sec"], 1e-6))
    warn_leak_origin = d["base_common"] or leaked

    if wi_key == "base":
        b_cls = "wi-banner warn" if warn_leak_origin else "wi-banner"
        extra_w = " 단, 이미 유출·흔한 목록에 포함된 비밀번호는 뒤에 글자만 덧붙이지 말고 기본 단어 자체를 변경해야 안전합니다." if warn_leak_origin else ""
        md(
            f'<div class="{b_cls}"><b>전후 비교 요약:</b> 현재 예상 버팀 시간은 <b>{human(base_sc["sec"])}</b>입니다. '
            f'여기에 <b>특수문자 1개</b>를 추가하면 <b>{human(scs["sp1"]["sec"])}</b>(약 ×{mult_sp:,.0f}배), '
            f'<b>특수문자 + 4자</b>를 추가하면 <b>{human(scs["both"]["sec"])}</b>으로 증가합니다.{extra_w}</div>'
        )
    else:
        b_cls = "wi-banner warn" if warn_leak_origin else "wi-banner"
        extra_w = " (원래 비밀번호가 유출·흔한 목록에 있었으므로 실제 변경 시에는 기본 단어도 함께 바꾸세요.)" if warn_leak_origin else ""
        md(
            f'<div class="{b_cls}"><b>실험 결과 ({act_sc["label"]}):</b> 예상 버팀 시간이 <b>{human(base_sc["sec"])}</b>에서 '
            f'<b>{human(act_sc["sec"])}</b>으로 <b>약 ×{mult_act:,.0f}배</b> 길어졌습니다.{extra_w}</div>'
        )

    c1, c2, c3 = st.columns([3.4, 3.4, 3.2])
    with c1:
        order_keys = ["base", "sp1", "len4", "both"]
        y_lbls = [f"{scs[k]['short']} ({scs[k]['L']}자)" for k in order_keys]
        secs = [scs[k]["sec"] for k in order_keys]
        l_vals = [float(np.clip(math.log10(max(v, 1e-3)), -3.0, 18.0)) for v in secs]
        bar_cols = [GREEN if v >= YR else (ORANGE if v >= 3600 else RED) for v in secs]
        b_widths = [3.0 if k == wi_key else 0 for k in order_keys]
        txt_lbl = [human(v).replace(" (1ms 미만)", "") for v in secs]
        t_pairs = [(0.2, "1ms"), (3.2, "1초"), (6.756, "1시간"), (10.699, "1년"), (12.699, "100년")]
        if max(l_vals) > 12.0:
            t_pairs.append((16.699, "10⁶년"))

        fig_wi = go.Figure(go.Bar(
            y=y_lbls, x=[v + 3.2 for v in l_vals], orientation="h",
            marker=dict(color=bar_cols, line=dict(color="#1d1d1f", width=b_widths)),
            text=txt_lbl, textposition="outside", cliponaxis=False,
            textfont=dict(size=12, weight="bold", color="#1d1d1f"),
            hovertemplate="%{y}: %{text}<extra></extra>"
        ))
        fig_wi.add_vline(x=math.log10(YR) + 3.2, line_dash="dot", line_color="#1d1d1f", line_width=1.8, annotation_text="기준 1년", annotation_position="top")
        fig_wi.update_layout(title=dict(text="<b>조건별 예상 버팀 시간 비교</b>", font=dict(size=13.5, color="#1d1d1f"), x=0.02, y=0.95))
        fig_wi.update_xaxes(
            tickvals=[p[0] for p in t_pairs],
            ticktext=[p[1] for p in t_pairs],
            range=[0, max(max(l_vals) + 3.2, math.log10(YR) + 3.2) + 4.8]
        )
        fig_wi.update_yaxes(autorange="reversed", showgrid=False)
        show(style(fig_wi, 238, t=42, l=10, r=24), f"d_wi_{wi_key}_{act_sc['L']}")

        vf = go.Figure([go.Violin(x=M[M.등급 == g_].length, name=g_, orientation="h", line_color=CCOL[g_], points=False, showlegend=False, spanmode="hard") for g_ in ORDER])
        if wi_key != "base" and act_sc["L"] != base_sc["L"]:
            vf.add_vline(x=base_sc["L"], line_color=GRAY, line_dash="dot", annotation_text=f"기존 {base_sc['L']}자", annotation_position="bottom left")
            vf.add_vline(x=act_sc["L"], line_color=BLUE, line_width=2.5, annotation_text=f"변경 {act_sc['L']}자", annotation_position="top right")
        else:
            vf.add_vline(x=act_sc["L"], line_color="#1d1d1f", line_dash="dash", line_width=2, annotation_text=f"내 길이 {act_sc['L']}자")
        vf.update_layout(title=dict(text="<b>25,000건 표본 등급별 길이 분포</b>", font=dict(size=13.5, color="#1d1d1f"), x=0.02, y=0.94))
        vf.update_xaxes(title="비밀번호 길이(자)")
        show(style(vf, 214, t=38, l=10, r=18), f"d_vf_{wi_key}_{act_sc['L']}")

    with c2:
        s = act_sc["score"]
        grade, g_col = ("위험", RED) if s < 30 else ("주의", ORANGE) if s < 55 else ("양호", BLUE) if s < 80 else ("안전", GREEN)
        if wi_key == "base" and (base_sc["common"] or leaked):
            grade, g_col = "위험", RED

        if base_sc["common"]:
            common_html = '<div class="hibp-bar hibp-bad"><span>📋 상위 500개 흔한 목록</span><span>포함됨 · 사전 공격 시 즉시 해독</span></div>'
            common_chip = ("흔한 목록 아님", False)
        else:
            common_html = '<div class="hibp-bar hibp-ok"><span>📋 상위 500개 흔한 목록</span><span>해당 없음</span></div>'
            common_chip = ("흔한 목록 아님", True)

        if hibp_cnt is None:
            hibp_html = '<div class="hibp-bar hibp-na"><span>🌐 HIBP 실제 유출 조회</span><span>조회 실패 · 확인 불가</span></div>'
            hibp_chip = ("유출 확인 불가", None)
        elif hibp_cnt > 0:
            hibp_html = f'<div class="hibp-bar hibp-bad"><span>🚨 HIBP 실제 유출 조회 (원본)</span><span>전 세계 {hibp_cnt:,}회 유출</span></div>'
            hibp_chip = ("유출 이력 있음", False)
        else:
            hibp_html = '<div class="hibp-bar hibp-ok"><span>🛡️ HIBP 실제 유출 조회 (원본)</span><span>0회 (유출 기록 없음)</span></div>'
            hibp_chip = ("유출 이력 없음", True)

        chips = [
            ("12자 이상", act_sc["L"] >= 12), ("대문자 포함", act_sc["up"]), ("숫자 포함", act_sc["dg"]),
            ("특수문자 포함", act_sc["sp"]), common_chip, hibp_chip,
            ("연속 문자 없음", not act_sc["seq"]), ("반복 문자 없음", not act_sc["rep"]), ("연도·날짜 없음", not act_sc["yr"])
        ]
        ch = "".join(
            f'<div class="chip {"na" if v is None else "ok" if v else "no"}">{"–" if v is None else "✓" if v else "✕"} {k}</div>'
            for k, v in chips
        )
        time_sub = "흔한 목록 포함 → 사전 공격 시 즉시" if act_sc["common"] else "조합 대입 + 패턴 보정 예측치"

        md(f"""<div class="res-card">
            <div>
                <div class="res-hdr">
                    <span>종합 보안 진단 결과</span>
                    <span class="grade-pill" style="background:{g_col}">{grade} · {round(s)}점</span>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-lbl">예상 버팀 시간</div>
                        <div class="stat-val">{human(act_sc['sec'])}</div>
                        <div class="stat-sub">{time_sub}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-lbl">표본 내 안전 순위</div>
                        <div class="stat-val">상위 {max(0.1, 100 - act_sc['pct']):.1f}%</div>
                        <div class="stat-sub">고유 25,000건 기준</div>
                    </div>
                </div>
                {common_html}
                {hibp_html}
            </div>
            <div>
                <div style="font-size:13px;font-weight:800;color:#1d1d1f;margin-bottom:5px">9대 보안 구성 체크리스트 {"(실험 반영)" if wi_key != "base" else ""}</div>
                <div class="chk-grid">{ch}</div>
            </div>
        </div>""")

    with c3:
        tips = []
        if leaked:
            tips.append(("warn", "실제 유출", f"HIBP 유출 데이터베이스에서 <b>{hibp_cnt:,}회</b> 확인되었습니다. 이미 외부에 공개된 비밀번호이므로 즉시 변경하세요."))
        if base_sc["common"]:
            tips.append(("warn", "흔한 비밀번호", "전 세계 상위 500위 안에 드는 흔한 비밀번호입니다. 연산 시간과 무관하게 사전 대입 공격으로 1초 내에 뚫립니다."))
        if hibp_cnt is None:
            tips.append(("norm", "조회 실패", "HIBP 서버에 연결하지 못해 유출 여부는 확인하지 못했습니다. 안전하다는 뜻이 아니니 잠시 후 다시 점검해 보세요."))
        if act_sc["L"] < 12:
            tips.append(("norm", "길이 보강", f"현재 길이(<b>{act_sc['L']}자</b>)는 다소 짧습니다. 최소 12자 이상으로 늘리면 대입해야 할 경우의 수가 기하급수적으로 증가합니다."))
        if not (act_sc["up"] and act_sc["dg"] and act_sc["sp"]):
            tips.append(("norm", "문자 조합", "영문 대소문자, 숫자, 특수문자를 고르게 섞으면 한 자리당 탐색해야 할 후보 문자 수가 크게 넓어집니다."))
        if act_sc["seq"] or act_sc["rep"] or act_sc["yr"]:
            tips.append(("norm", "패턴 제거", "키보드 연속 배열(qwerty, 1234), 동일 문자 반복(aaa), 연도·생일(2024, 0315)은 공격 도구가 우선 대입하는 패턴입니다."))
        if not tips:
            tips.append(("good", "최상 등급", "유출 기록이 없고 길이와 문자 다양성 모두 우수합니다. 주요 계정에는 2단계 인증(패스키·OTP)을 함께 설정해 두세요."))

        tips.append((
            "norm", "데이터 검증",
            "방금 실험한 <b>길이와 문자 조합의 방어 효과</b>, 그리고 <b>흔한 비밀번호의 위험성</b>을 상단 <b>2~4번 탭</b>의 25,500건 공개 데이터 차트로 검증해 보세요."
        ))

        tip_html = "".join(f'<div class="tip-item {cls}"><span class="tip-tag">{tag}</span>{txt}</div>' for cls, tag, txt in tips[:4])
        md(f"""<div class="res-card" style="justify-content:space-between">
            <div>
                <div class="res-hdr"><span>맞춤 보안 처방 가이드</span></div>
                <div>{tip_html}</div>
            </div>
        </div>""")

    nb = st.columns([7.4, 2.6])
    nb[1].button("다음 · 2. 공격 방식과 속도 비교 →", on_click=_go_nav, args=(NAMES[1],), **_BTN_W)
    source("t3")

def p_intro():
    head(
        "질문 2. 공격 방식에 따라 비밀번호가 뚫리는 속도는 얼마나 달라질까?",
        "공개 데이터 25,500건을 바탕으로 로그인 창 직접 대입(초당 10회)과 유출 파일 고속 대입(초당 10억 회)의 위험도를 비교합니다."
    )
    fl = filters(); TS = {"1분": 60, "1시간": 3600, "1일": 86400, "1년": YR, "100년": YR * 100}
    tk = fl[0].select_slider("이 시간 안에 뚫리면 '위험'으로 집계합니다", options=list(TS), value="1년", key="i_t"); T = TS[tk]
    Ln = fl[1].slider("비교할 내 비밀번호 길이(자)", 4, 24, 8, key="i_l")
    fl[2].caption("↻ 기준 시간과 길이를 바꾸면 아래 지표와 차트 3개가 함께 바뀝니다")

    pc = [(M.crack_time_sec <= T).mean() * 100, (P.online_sec <= T).mean() * 100, (P.offline_crack_sec <= T).mean() * 100]
    ins(
        f"같은 <b>{tk}</b> 기준이라도 <b>로그인 창 직접 대입(초당 10회)</b>으로는 흔한 비밀번호의 <b>{pc[1]:.0f}%</b>가 뚫리지만, "
        f"유출된 암호화 파일을 장비로 돌리는 <b>유출 파일 고속 대입(초당 10억 회, 1억 배 빠름)</b>으로는 <b>{pc[2]:.0f}%</b>가 뚫립니다. "
        f"게다가 흔한 목록부터 먼저 넣어보는 <b>사전 대입 공격</b> 앞에서는 상위 500개 모두 계산 시간과 무관하게 <b>1초 이내 즉시</b> 해독됩니다.",
        GREEN, "공격 방식별 차이"
    )

    gap = float((P.lon - P.lof).median())
    slp = float(np.polyfit(M.length, M.lc, 1)[0])
    infographic([
        ("📦", f"{len(M) + len(P):,}", "개", f"분석 데이터 (고유 피처 {len(M):,}개 + 흔한 비밀번호 {len(P)}개)"),
        ("🔓", f"{pc[1]:.0f}", "%", f"흔한 비밀번호 중 로그인 창 대입으로 {tk} 안에 뚫리는 비율"),
        ("⚡", f"10{sup(round(gap))}", "배", "유출 파일 고속 대입(초당 10억 회)이 로그인 창 대입보다 빠른 배율"),
        ("📏", f"×{10 ** slp:.0f}", "배 / 1자", "비밀번호가 1자 길어질 때 버팀 시간이 늘어나는 평균 배율")
    ])

    x1, x2, x3 = st.columns([3.5, 3.2, 3.3])
    with x1:
        sub_l = M[M.length.between(Ln - 2, Ln + 2)]
        pl = float((sub_l.crack_time_sec <= T).mean() * 100) if len(sub_l) else 0.0
        wf = [pl, pc[1], pc[2]]
        head(
            "100개 중 몇 개가 위험할까? (와플 차트)",
            f"빨간 칸 1개 = {tk} 안에 뚫리는 비율 1%입니다. 내 길이({Ln}자±2자) 표본과 흔한 비밀번호의 공격 방식별 위험 비율을 비교합니다."
        )
        ttl = [f"내 길이({Ln}자±2)", "로그인 창 대입", "유출 파일 대입"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t}<br><b>{p:.0f}%</b>" for t, p in zip(ttl, wf)])
        for i, p in enumerate(wf):
            z = np.zeros(100); z[:int(round(p))] = 1
            fig.add_trace(go.Heatmap(z=z.reshape(10, 10), colorscale=[[0, "#cfeedd"], [1, "#ff6b6b"]], showscale=False, xgap=3, ygap=3, hoverinfo="skip"), 1, i + 1)
            fig.update_xaxes(visible=False, row=1, col=i + 1); fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i + 1)
        fig.update_annotations(font_size=11)
        show(style(fig, 310, t=64).update_layout(margin=dict(l=12, r=12, t=64, b=12)), f"i1_{tk}_{Ln}")

    with x2:
        def _med_sec(cond, fb_len):
            s_ = M.loc[cond, "crack_time_sec"]
            if len(s_) < 10:
                s_ = M.loc[M.length == fb_len, "crack_time_sec"]
            if s_.empty:
                s_ = M.loc[M.length.between(fb_len - 1, fb_len + 1), "crack_time_sec"]
            return float(s_.median())

        ln_med = _med_sec(M.length == Ln, Ln)
        ln_str = human(ln_med).replace(" (1ms 미만)", "")
        verdict = "기준보다 오래 버팁니다" if ln_med > T else "기준 안에 뚫립니다"
        head(
            "특수문자 포함 vs 글자 수 증가 비교 (막대 차트)",
            f"선택한 {Ln}자의 중앙값은 {ln_str}으로 기준({tk})보다 {verdict}. 점선 오른쪽(초록)이 기준 이상 버티는 구간입니다."
        )
        groups = [
            ("8자 · 특수문자 포함", _med_sec((M.length == 8) & (M.n_special > 0), 8)),
            ("10자 · 특수문자 없음", _med_sec((M.length == 10) & (M.n_special == 0), 10)),
            ("12자 · 특수문자 없음", _med_sec((M.length == 12) & (M.n_special == 0), 12)),
            ("16자 · 특수문자 없음", _med_sec((M.length == 16) & (M.n_special == 0), 16)),
            (f"내 길이 {Ln}자", ln_med),
        ]
        y_lbls = [g[0] for g in groups]
        meds = [g[1] for g in groups]
        cols = [GREEN if v > T else RED for v in meds]
        l_meds = [float(np.clip(math.log10(max(v, 1e-3)), -3, 18)) for v in meds]
        txt_lbl = [human(v).replace(" (1ms 미만)", "") for v in meds]
        t_pairs = [(0.2, "1ms"), (3.2, "1초"), (6.756, "1시간"), (10.699, "1년"), (12.699, "100년")]
        if max(l_meds) > 12.0:
            t_pairs.append((16.699, "10⁶년"))
        fig = go.Figure(go.Bar(
            y=y_lbls, x=[v + 3.2 for v in l_meds], orientation="h",
            marker=dict(color=cols, line=dict(color="#1d1d1f", width=[0, 0, 0, 0, 3])),
            text=txt_lbl, textposition="outside", cliponaxis=False,
            textfont=dict(size=12, weight="bold", color="#1d1d1f"),
            hovertemplate="%{y}: %{text}<extra></extra>"
        ))
        fig.add_vline(x=math.log10(T) + 3.2, line_dash="dot", line_color="#1d1d1f", line_width=2, annotation_text=f"기준 {tk}", annotation_position="top")
        fig.update_xaxes(
            tickvals=[p[0] for p in t_pairs],
            ticktext=[p[1] for p in t_pairs],
            range=[0, max(max(l_meds), math.log10(T)) + 6.5]
        )
        fig.update_yaxes(showgrid=False)
        show(style(fig, 310, t=34, l=12, r=28), f"i2_{tk}_{Ln}")

    with x3:
        head(
            "길이별 버팀 시간 증가 추세 (회귀 추세선)",
            "파란 점은 길이별 중앙값, 검은 점선은 지수 회귀 추세선입니다. 빨간 기준선(1년) 위에 머무르려면 최소 11~12자 이상이 필요합니다."
        )
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
        show(style(fig, 310, t=34), f"i3_{Ln}")

    nb = st.columns([7.4, 2.6])
    nb[1].button("다음 · 3. 길이와 문자 조합 검증 →", on_click=_go_nav, args=(NAMES[2],), **_BTN_W)
    with st.expander("데이터 출처 · 시간(초) 계산 기준 요약"):
        st.markdown(
            "- **Password Metrics Dataset (고유 25,000건):** 문자 풀 크기($N$)와 길이($L$)의 경우의 수($N^L$) 기반 무차별 대입 시간(초). 등급별 약 5,000건씩 추출된 균등 표본 — [Kaggle 바로가기](https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset)\n"
            "- **Password Strength Dataset (흔한 500건):** 로그인 창 대입(초당 10회) vs 유출 파일 대입(초당 10억 회) 시간 비교. 실제 사전 공격 시에는 모두 즉시 해독됨 — [Kaggle 바로가기](https://www.kaggle.com/datasets/utkarshx27/passwords)"
        )

def p_leak():
    head(
        "질문 3. 비밀번호 길이와 문자 조합을 바꾸면 얼마나 오래 버틸까?",
        "고유 비밀번호 피처 25,000건의 경우의 수(N^L) 기반 버팀 시간을 분석합니다. (원본은 등급별 약 5,000건씩 추출한 균등 표본입니다)"
    )
    fl = filters()
    gs = multi_pick(fl[0], "강도 등급", ORDER, "a_g")
    lr = fl[1].slider("길이 범위(자)", 4, 63, (4, 40), key="a_l")
    fl[2].caption("↻ 필터를 바꾸면 상단 지표와 차트 3개가 함께 바뀝니다")

    f = M[M.등급.isin(gs) & M.length.between(*lr)]
    if f.empty:
        st.info("조건에 맞는 데이터가 없습니다. 등급이나 길이 범위를 넓혀 보세요.")
        return

    ent_col = "entropy" if "entropy" in f.columns else "shannon"
    kpis([
        ("선택된 표본 수", len(f), 0, "개"),
        ("길이 중앙값", float(f.length.median()), 0, "자"),
        ("문자 복잡도 (엔트로피)", float(f[ent_col].median()), 1, "bit"),
        ("버팀 시간 중앙값", human(f.crack_time_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("1년 내 뚫리는 비율", (f.crack_time_sec <= YR).mean() * 100, 0, "%"),
        ("1시간 내 뚫리는 비율", (f.crack_time_sec <= 3600).mean() * 100, 0, "%")
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
            f"길이가 1자 늘 때마다 버팀 시간이 평균 약 {10**slope:.1f}배 늘어납니다(선형 설명력 R²={r2:.2f}). 표본 20개 미만 길이는 제외했습니다."
            if np.isfinite(slope) else "길이 구간을 넓히면 추세가 보입니다."
        )
        tbl_a1 = (lambda: st.dataframe(g.assign(중앙값=g.med.map(human))[["length", "n", "중앙값"]].rename(columns={"length": "길이", "n": "표본수"}), hide_index=True)) if not g.empty else None
        head("길이가 길어질수록 얼마나 오래 버틸까? (선 그래프)", desc, table_fn=tbl_a1)
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
        stp = [("전체 시작", 0), ("1분 이상 버팀", 60), ("1시간 이상 버팀", 3600), ("1일 이상 버팀", 86400), ("1년 이상 버팀", YR), ("100년 이상 버팀", YR * 100)]
        vals = [100.0] + [(f.crack_time_sec > t).mean() * 100 for _, t in stp[1:]]
        head("시간대별로 표본의 몇 %가 남을까? (깔때기 차트)", f"선택한 표본 중 {vals[3]:.0f}%가 1일을, {vals[4]:.0f}%가 1년을 버팁니다.")
        fig = go.Figure(go.Funnel(
            y=[a for a, _ in stp], x=vals, text=[f"{v:.0f}%" for v in vals], textinfo="text", textposition="inside",
            textfont=dict(size=13, color="#fff"),
            marker=dict(color=["#5b8def", "#6e9bf0", "#4cc38a", "#3bb27a", "#2a9d68", "#1f8a58"]),
            connector=dict(line=dict(color="#e5e7eb"))
        ))
        show(style(fig), "a2")

    with x3:
        head("문자 조합 유형별 버팀 시간 분포 (박스 플롯)", "숫자만 쓴 경우보다 영문·숫자·특수문자를 함께 섞을수록 버팀 시간 분포 전체가 크게 상승합니다.")
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

    ins(
        f"선택한 표본 {len(f):,}개 중 <b>{(f.crack_time_sec <= YR).mean():.0%}</b>가 모든 조합 대입 기준 1년 안에 뚫립니다(길이 중앙값 {int(f.length.median())}자 · 버팀 중앙값 {human(f.crack_time_sec.median())}). "
        f"단, 이 수치는 경우의 수(Nᴸ) 이론치이므로 연도·키보드 배열 패턴이 포함되면 실제 방어 시간은 이보다 훨씬 짧아집니다."
    )
    nb = st.columns([7.4, 2.6])
    nb[1].button("다음 · 4. 흔한 비밀번호 500건 검증 →", on_click=_go_nav, args=(NAMES[3],), **_BTN_W)
    source("t1")

def p_common():
    head(
        "질문 4. 계산상 '7년 버틴다'는 흔한 비밀번호가 왜 실제로는 1초 만에 뚫릴까?",
        "상위 500개 흔한 비밀번호 데이터로 로그인 창 대입(초당 10회)과 유출 파일 대입(초당 10억 회), 사전 공격의 차이를 확인합니다."
    )
    fl = filters()
    n = fl[0].slider("상위 몇 개를 볼까요?", 5, 500, 20, 5, key="b_n")
    allc = sorted(P.category.unique())
    cats = multi_pick(fl[1], "비밀번호 유형", allc, "b_c")
    fl[2].caption("↻ 순위와 유형을 바꾸면 지표와 차트가 함께 바뀝니다")

    q = P[(P["rank"] <= n) & P.category.isin(cats)]
    if q.empty:
        st.info("조건에 맞는 비밀번호가 없습니다. 유형을 선택해 주세요.")
        return

    topc = q.category.value_counts()
    kpis([
        ("표시 중 비밀번호", len(q), 0, "개"),
        ("가장 많은 유형", f"{topc.index[0]}", 0, f" {topc.iloc[0]/len(q):.0%}"),
        ("로그인 창 대입 · 중앙값", human(q.online_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("유출 파일 대입 · 중앙값", human(q.offline_crack_sec.median()).replace(" (1ms 미만)", ""), 0, ""),
        ("로그인 창 · 1일 내 뚫림", (q.online_sec <= 86400).mean() * 100, 0, "%"),
        ("유출 파일 · 1초 내 뚫림", (q.offline_crack_sec <= 1).mean() * 100, 0, "%")
    ])

    x1, x2, x3 = st.columns([4, 3, 3])
    with x1:
        tbl_b1 = lambda: st.dataframe(
            q.assign(로그인창_대입=q.online_sec.map(human), 유출파일_대입=q.offline_crack_sec.map(human))[["rank", "password", "category", "로그인창_대입", "유출파일_대입"]]
            .rename(columns={"rank": "순위", "password": "비밀번호", "category": "유형"}),
            hide_index=True
        )
        head(
            f"상위 {len(q)}개 흔한 비밀번호의 '보안 착시' (타일 차트)",
            "로그인 창 무차별 대입 기준 색상입니다. 글자 수가 길어 초록색(수년)으로 보여도 유출 파일 대입이나 사전 공격 앞에서는 즉시 뚫리는 착시입니다.",
            table_fn=tbl_b1
        )
        tile_grid(q)

    with x2:
        head("흔한 비밀번호는 어떤 유형이 가장 많을까? (도넛 차트)", f"가장 많은 유형은 '{topc.index[0]}'({topc.iloc[0]}개)입니다. 기억하기 쉬운 단어·이름·숫자 배열이 대부분을 차지합니다.")
        fig = go.Figure(go.Pie(labels=topc.index, values=topc.values, hole=.58, sort=False, textinfo="percent", marker=dict(colors=px.colors.qualitative.Set3 * 2, line=dict(color="#fff", width=2))))
        fig.add_annotation(text=f"<b>{len(q)}개</b>", x=.5, y=.5, showarrow=False, font=dict(size=22))
        show(style(fig, 300).update_layout(legend=dict(orientation="v", x=1.0, y=.5, font=dict(size=11))), "b2")

    with x3:
        a_, b_ = (q.online_sec <= 86400).mean() * 100, (q.offline_crack_sec <= 86400).mean() * 100
        head("로그인 창 대입 vs 유출 파일 대입 속도 비교 (누적 차트)", f"1일 안에 로그인 창 대입은 {a_:.0f}%, 유출 파일 대입은 {b_:.0f}%가 뚫립니다. 대입 속도가 1억 배 빨라 곡선이 왼쪽으로 이동합니다.")
        fig = go.Figure()
        for nm, col, rgba, v in [
            ("로그인 창 대입 (초당 10회)", BLUE, "rgba(91,141,239,.18)", q.lon),
            ("유출 파일 대입 (초당 10억 회)", RED, "rgba(255,59,48,.14)", q.lof)
        ]:
            xs = np.sort(v.values); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
            fig.add_trace(go.Scatter(x=xs, y=ys, name=nm, mode="lines", line=dict(color=col, width=3, shape="hv"), fill="tozeroy", fillcolor=rgba, hovertemplate="%{y:.0f}%가 뚫림<extra>" + nm + "</extra>"))
        tick_axis(fig, min(q.lon.min(), q.lof.min()), max(q.lon.max(), q.lof.max()), "x")
        fig.update_yaxes(title="뚫린 비율(%)", range=[0, 102])
        show(style(fig), "b3")

    ins(
        f"1위 <b>'password'</b>도 로그인 창 무차별 대입(초당 10회)으로는 <b>7년</b>이 걸린다고 계산되지만, "
        f"유출 파일 고속 대입(초당 10억 회)으로는 <b>2초</b>, 흔한 목록부터 대조하는 <b>사전 대입 공격으로는 0.001초 미만</b>에 즉시 뚫립니다. "
        f"유출 목록에 없는 조합으로 <b>12자 이상</b> 길게 만드는 것이 핵심입니다.",
        RED, "보안 착시 주의"
    )
    nb = st.columns([7.4, 2.6])
    nb[1].button("처음으로 · 1. 내 비밀번호 실험하기 →", on_click=_go_nav, args=(NAMES[0],), **_BTN_W)
    source("t2")

if page == 0: p_lab()
elif page == 1: p_intro()
elif page == 2: p_leak()
else: p_common()
