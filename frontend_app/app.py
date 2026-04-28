"""
Single-page aviation safety intelligence dashboard.
Navigation uses in-page anchors with smooth scrolling (no separate Streamlit pages).
"""

import streamlit as st

from utils.dashboard_sections import (
    render_cluster_map,
    render_overview,
    render_research_chatbot,
    render_research,
    render_time_series,
)
from utils.data_loader import load_data_bundle

st.set_page_config(
    page_title="Aviation Safety Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_DASH_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;600;700;800&display=swap');

  html { scroll-behavior: smooth; }
  .stApp {
    position: relative;
    background: radial-gradient(1200px 600px at 10% -10%, rgba(56, 139, 253, 0.12), transparent 55%),
                radial-gradient(900px 500px at 100% 0%, rgba(0, 212, 170, 0.08), transparent 50%),
                linear-gradient(180deg, #06080c 0%, #0a0e14 40%, #0d1117 100%) !important;
    color: #e6edf3;
    font-family: "DM Sans", system-ui, sans-serif;
    width: 100vw !important;
    max-width: 100vw !important;
  }
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main {
    width: 100vw !important;
    max-width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  [data-testid="stAppViewContainer"] > .main {
    padding-left: 0 !important;
    padding-right: 0 !important;
  }
  .block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 0.35rem !important;
    padding-right: 0.35rem !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
  }

  /* --- Night sky: twinkling stars (behind UI) --- */
  .stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
      radial-gradient(1px 1px at 8% 12%, rgba(255,255,255,0.95), transparent),
      radial-gradient(1px 1px at 22% 8%, rgba(230,240,255,0.85), transparent),
      radial-gradient(1px 1px at 38% 24%, rgba(255,255,255,0.75), transparent),
      radial-gradient(1px 1px at 52% 6%, rgba(255,255,255,0.9), transparent),
      radial-gradient(1px 1px at 68% 18%, rgba(220,235,255,0.8), transparent),
      radial-gradient(1px 1px at 82% 11%, rgba(255,255,255,0.85), transparent),
      radial-gradient(1px 1px at 94% 28%, rgba(255,255,255,0.7), transparent),
      radial-gradient(1px 1px at 14% 38%, rgba(255,255,255,0.8), transparent),
      radial-gradient(1px 1px at 31% 44%, rgba(200,220,255,0.75), transparent),
      radial-gradient(1px 1px at 47% 36%, rgba(255,255,255,0.9), transparent),
      radial-gradient(1px 1px at 61% 48%, rgba(255,255,255,0.65), transparent),
      radial-gradient(1px 1px at 76% 41%, rgba(255,255,255,0.85), transparent),
      radial-gradient(1px 1px at 89% 52%, rgba(230,240,255,0.7), transparent),
      radial-gradient(1px 1px at 6% 58%, rgba(255,255,255,0.8), transparent),
      radial-gradient(1px 1px at 19% 66%, rgba(255,255,255,0.75), transparent),
      radial-gradient(1px 1px at 35% 72%, rgba(255,255,255,0.9), transparent),
      radial-gradient(1px 1px at 54% 63%, rgba(220,235,255,0.7), transparent),
      radial-gradient(1px 1px at 71% 74%, rgba(255,255,255,0.85), transparent),
      radial-gradient(1px 1px at 86% 68%, rgba(255,255,255,0.65), transparent),
      radial-gradient(1px 1px at 11% 84%, rgba(255,255,255,0.8), transparent),
      radial-gradient(1px 1px at 28% 91%, rgba(255,255,255,0.7), transparent),
      radial-gradient(1px 1px at 44% 88%, rgba(230,240,255,0.85), transparent),
      radial-gradient(1px 1px at 63% 93%, rgba(255,255,255,0.75), transparent),
      radial-gradient(1px 1px at 79% 86%, rgba(255,255,255,0.9), transparent),
      radial-gradient(1px 1px at 92% 78%, rgba(255,255,255,0.65), transparent),
      radial-gradient(1.5px 1.5px at 41% 15%, rgba(88,166,255,0.5), transparent),
      radial-gradient(1.5px 1.5px at 73% 55%, rgba(0,212,170,0.45), transparent),
      radial-gradient(1.5px 1.5px at 17% 71%, rgba(88,166,255,0.4), transparent);
    background-repeat: repeat;
    background-size: 420px 380px;
    animation: skyDrift 90s linear infinite, starTwinkle 5.5s ease-in-out infinite alternate;
  }
  @keyframes skyDrift {
    from { background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0; }
    to { background-position: 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px, 420px 120px; }
  }
  @keyframes starTwinkle {
    from { opacity: 0.42; }
    to { opacity: 0.78; }
  }

  /* Keep all real UI above pseudo-layers (::after otherwise paints over .stApp children). */
  [data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 2;
  }

  .stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
      radial-gradient(2px 2px at 25% 30%, rgba(255,255,255,0.35), transparent),
      radial-gradient(2px 2px at 60% 22%, rgba(88,166,255,0.25), transparent),
      radial-gradient(2px 2px at 48% 70%, rgba(255,255,255,0.3), transparent),
      radial-gradient(2px 2px at 85% 60%, rgba(0,212,170,0.2), transparent),
      radial-gradient(2px 2px at 12% 55%, rgba(255,255,255,0.28), transparent),
      radial-gradient(2px 2px at 72% 88%, rgba(255,255,255,0.22), transparent);
    background-size: 100% 100%;
    animation: bigStarPulse 9s ease-in-out infinite alternate;
  }
  @keyframes bigStarPulse {
    from { opacity: 0.25; }
    to { opacity: 0.5; }
  }

  .flight-layer {
    position: fixed;
    inset: 0;
    /* Must stay below Streamlit widgets (z-index:auto = 0); z-index:1 was covering all content. */
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
  }
  .flight-layer .plane {
    position: absolute;
    left: -12%;
    color: rgba(196, 224, 255, 0.95);
    font-size: clamp(1.45rem, 3.1vw, 2.8rem);
    line-height: 1;
    text-shadow: 0 0 18px rgba(88, 166, 255, 0.62), 0 0 30px rgba(0, 212, 170, 0.24);
    opacity: 0.72;
    animation-name: planeCruise;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    will-change: transform;
  }
  .flight-layer .plane svg {
    display: block;
    width: 2.2em;
    height: auto;
    fill: currentColor;
  }
  @keyframes planeCruise {
    0% { transform: translate3d(0, 0, 0) rotate(-6deg) scale(1); }
    100% { transform: translate3d(125vw, -1.5vh, 0) rotate(-6deg) scale(1); }
  }
  .flight-layer .p1 { top: 11%; animation-duration: 42s; animation-delay: -8s; animation-name: planeCruiseRev; }
  .flight-layer .p2 { top: 26%; animation-duration: 58s; animation-delay: -22s; opacity: 0.54; }
  .flight-layer .p3 { top: 52%; animation-duration: 36s; animation-delay: -4s; animation-name: planeCruiseRev; opacity: 0.58; font-size: clamp(1.1rem, 2.1vw, 1.8rem); }
  .flight-layer .p4 { top: 72%; animation-duration: 68s; animation-delay: -35s; opacity: 0.48; font-size: clamp(1.4rem, 3vw, 2.7rem); }
  .flight-layer .p5 { top: 38%; animation-duration: 48s; animation-delay: -15s; opacity: 0.6; font-size: clamp(1.2rem, 2.4vw, 2.1rem); }
  .flight-layer .p6 { top: 84%; animation-duration: 56s; animation-delay: -28s; animation-name: planeCruiseRev; opacity: 0.52; font-size: clamp(1rem, 2vw, 1.7rem); }
  @keyframes planeCruiseRev {
    0% { transform: translate3d(0, 0, 0) rotate(8deg) scaleX(-1); }
    100% { transform: translate3d(125vw, 2vh, 0) rotate(8deg) scaleX(-1); }
  }
  [data-testid="stHeader"] {
    position: relative;
    z-index: 1000002;
    background: rgba(6, 8, 12, 0.75);
    border-bottom: 1px solid #21262d;
    backdrop-filter: blur(12px);
  }
  [data-testid="stToolbar"] { display: none; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  [data-testid="stSidebar"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }

  .block-container {
    position: relative;
    z-index: 1;
    padding-top: 1.25rem !important;
    padding-bottom: 4rem !important;
    max-width: none !important;
    width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  .nav-wrap {
    position: sticky;
    top: 0.35rem;
    z-index: 1000000;
    margin: 0 0 1.75rem 0;
  }
  .nav-inner {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    padding: 0;
    overflow: hidden;
    background: linear-gradient(160deg, rgba(22, 28, 38, 0.97) 0%, rgba(8, 11, 16, 0.99) 55%, rgba(12, 18, 28, 0.98) 100%);
    border: 1px solid rgba(88, 166, 255, 0.22);
    border-radius: 20px;
    box-shadow:
      0 16px 48px rgba(0, 0, 0, 0.5),
      0 0 0 1px rgba(0, 212, 170, 0.06) inset,
      0 1px 0 rgba(255, 255, 255, 0.06) inset;
    backdrop-filter: blur(16px);
  }
  .brand-masthead {
    position: relative;
    text-align: center;
    padding: 1.35rem 1.25rem 1.15rem;
    isolation: isolate;
  }
  .brand-masthead::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    background:
      radial-gradient(ellipse 85% 120% at 50% -20%, rgba(88, 166, 255, 0.18), transparent 55%),
      radial-gradient(ellipse 70% 80% at 80% 120%, rgba(0, 212, 170, 0.08), transparent 45%);
    pointer-events: none;
  }
  .brand-masthead::after {
    content: "";
    position: absolute;
    left: 50%;
    bottom: 0;
    transform: translateX(-50%);
    width: min(320px, 72vw);
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, rgba(88, 166, 255, 0.65), rgba(0, 212, 170, 0.55), transparent);
    opacity: 0.85;
    animation: mastheadPulse 4.5s ease-in-out infinite;
  }
  @keyframes mastheadPulse {
    0%, 100% { opacity: 0.55; filter: blur(0); }
    50% { opacity: 1; filter: blur(0.5px); }
  }
  .brand-heading {
    margin: 0;
    font-family: "Outfit", "DM Sans", system-ui, sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 6.2vw, 3.35rem);
    letter-spacing: -0.045em;
    line-height: 1.05;
    background: linear-gradient(
      118deg,
      #ffffff 0%,
      #c8e4ff 22%,
      #58a6ff 42%,
      #7ee8c6 68%,
      #79c0ff 100%
    );
    background-size: 220% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 20px rgba(88, 166, 255, 0.28)) drop-shadow(0 4px 32px rgba(0, 212, 170, 0.12));
    animation: brandShimmer 12s ease-in-out infinite;
  }
  @keyframes brandShimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
  }
  .nav-links {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 6px;
    padding: 14px 18px 16px;
    margin: 0;
    background: rgba(6, 9, 14, 0.55);
    border-top: 1px solid rgba(48, 54, 61, 0.65);
  }
  .nav-links a {
    color: #8b949e;
    text-decoration: none !important;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 14px;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: color 0.2s, background 0.2s, border-color 0.2s;
  }
  .nav-links a:hover {
    color: #58a6ff;
    background: rgba(56, 166, 255, 0.1);
    border-color: rgba(56, 166, 255, 0.25);
  }

  .section-anchor { scroll-margin-top: 96px; height: 1px; margin: 0; padding: 0; border: 0; }

  .section-shell {
    margin-bottom: 3rem;
    padding: 28px 28px 32px;
    background: rgba(18, 24, 31, 0.82);
    border: 1px solid #30363d;
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
  }
  .alert-shell {
    background: linear-gradient(135deg, rgba(25, 14, 36, 0.88) 0%, rgba(14, 19, 31, 0.9) 100%);
    border-color: rgba(139, 92, 246, 0.45);
    box-shadow: 0 8px 26px rgba(88, 64, 140, 0.28);
  }
  /* Floating chatbot launcher (bottom-right corner) */
  div[data-testid="stPopover"],
  section.main div[data-testid="stPopover"],
  div[data-testid="stPopover"]:has(button) {
    position: fixed !important;
    right: 18px !important;
    bottom: 18px !important;
    left: auto !important;
    top: auto !important;
    z-index: 10000 !important;
    width: auto !important;
    max-width: none !important;
  }
  div[data-testid="stPopover"] button {
    border-radius: 999px !important;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.96) 0%, rgba(37, 24, 58, 0.96) 100%) !important;
    border: 1px solid rgba(88, 166, 255, 0.45) !important;
    color: #e6edf3 !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(88,166,255,0.18) inset !important;
    width: auto !important;
    height: 48px !important;
    min-width: 132px !important;
    min-height: 48px !important;
    padding: 0 16px !important;
    font-size: 1rem !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
  }
  div[data-testid="stPopover"] button:hover {
    border-color: rgba(139, 92, 246, 0.65) !important;
    transform: translateY(-1px);
  }
  div[data-baseweb="popover"] {
    width: min(420px, 92vw) !important;
    max-height: min(78vh, 760px) !important;
    border-radius: 14px !important;
    overflow: auto !important;
    border: 1px solid rgba(88,166,255,0.28) !important;
    background: linear-gradient(160deg, rgba(10,15,24,0.98) 0%, rgba(18,14,30,0.98) 100%) !important;
    box-shadow: 0 18px 42px rgba(0,0,0,0.52) !important;
  }
  .section-head {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .section-kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #58a6ff;
    font-weight: 600;
  }
  .section-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
  }
  .section-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 6px 0 20px 0;
    max-width: 720px;
    line-height: 1.5;
  }

  div[data-testid="stMetric"] {
    background: rgba(13, 17, 23, 0.9) !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
  }
  [data-testid="stMetricValue"] {
    color: #f0f6fc !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    line-height: 1.15 !important;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(13,17,23,0.6);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #30363d;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #8b949e !important;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(56, 166, 255, 0.15) !important;
    color: #58a6ff !important;
  }

  .signal-hero {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 8px 0 12px 0;
  }
  .signal-priority, .signal-z, .signal-conf {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid #30363d;
    font-family: "JetBrains Mono", monospace;
  }
  .signal-priority { color: #ff9f43; border-color: rgba(255,159,67,0.35); background: rgba(255,159,67,0.08); }
  .signal-z { color: #00d4aa; border-color: rgba(0,212,170,0.35); background: rgba(0,212,170,0.08); }
  .signal-conf { color: #58a6ff; border-color: rgba(88,166,255,0.35); background: rgba(88,166,255,0.08); }

  .intel-card {
    background: rgba(6, 8, 12, 0.65);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #c9d1d9;
  }

  .stDataFrame { border: 1px solid #30363d !important; border-radius: 12px !important; overflow: hidden; }
  .stExpander { border: 1px solid #30363d !important; border-radius: 12px !important; background: rgba(13,17,23,0.5) !important; }
</style>
"""

st.markdown(_DASH_CSS, unsafe_allow_html=True)

_PLANE_SVG = "✈"

st.markdown(
    f"""
<div class="flight-layer" aria-hidden="true">
  <div class="plane p1">{_PLANE_SVG}</div>
  <div class="plane p2">{_PLANE_SVG}</div>
  <div class="plane p3">{_PLANE_SVG}</div>
  <div class="plane p4">{_PLANE_SVG}</div>
  <div class="plane p5">{_PLANE_SVG}</div>
  <div class="plane p6">{_PLANE_SVG}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="nav-wrap">
  <div class="nav-inner">
    <header class="brand-masthead">
      <h1 class="brand-heading">SafetySignal</h1>
    </header>
    <nav class="nav-links" aria-label="Primary">
      <a href="#sec-overview">Overview</a>
      <a href="#sec-cluster-map">Cluster map</a>
      <a href="#sec-time-series">Trend</a>
      <a href="#sec-research">Alert</a>
    </nav>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

bundle = load_data_bundle()

# --- Hero strip (centered trio) ---
_pad_l, h1, h2, h3, _pad_r = st.columns([2, 2, 2, 2, 2])
with h1:
    st.metric("Reports", f"{len(bundle.clustered_reports):,}")
with h2:
    st.metric("Clusters", f"{bundle.cluster_summary['cluster'].nunique():,}")
with h3:
    st.metric("Trend rows", f"{len(bundle.monthly_cluster_trends):,}")

st.markdown("---")
# Floating popup launcher (bottom-right) for research assistant.
with st.popover("Research"):
    render_research_chatbot(bundle)

# --- Overview ---
st.markdown('<div id="sec-overview" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown(
    """<div class="section-head"><h2 class="section-title">Overview</h2></div>""",
    unsafe_allow_html=True,
)
render_overview(bundle)
st.markdown("</div>", unsafe_allow_html=True)

# --- Cluster map ---
st.markdown('<div id="sec-cluster-map" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown(
    """<div class="section-head"><h2 class="section-title">Cluster map</h2></div>
<p class="section-sub">Interactive UMAP projection with month, cluster, and vessel filters. Noise points use HDBSCAN label −1.</p>""",
    unsafe_allow_html=True,
)
render_cluster_map(bundle)
st.markdown("</div>", unsafe_allow_html=True)

# --- Trend ---
st.markdown('<div id="sec-time-series" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown(
    """<div class="section-head"><h2 class="section-title">Trend</h2></div>""",
    unsafe_allow_html=True,
)
render_time_series(bundle)
st.markdown("</div>", unsafe_allow_html=True)

# --- Alert (alerts + RAG + bulletins) ---
st.markdown('<div id="sec-research" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-shell alert-shell">', unsafe_allow_html=True)
st.markdown(
    """<div class="section-head"><h2 class="section-title">Alert</h2></div>
<p class="section-sub">Signal lab: each spike opens as its own tab with causal narrative, FAA RAG citations, and SMS checklists — plus optional bulletin export.</p>""",
    unsafe_allow_html=True,
)
render_research(bundle)
st.markdown("</div>", unsafe_allow_html=True)
