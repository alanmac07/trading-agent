"""
Trading Agent Swarm — Premium Live Demo
Dark glassmorphism UI | Top icon navigation | Infinite simulation
"""

import time
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from evolution import EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Trading Agent Swarm",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*  { box-sizing: border-box; margin: 0; }

html, body, [class*="st-"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

.stApp {
    background: radial-gradient(ellipse 100% 60% at 50% -10%, #0d2040 0%, #060d1b 55%) !important;
    min-height: 100vh;
}

/* ── hide default chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"]   { display: none; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton              { display: none; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebarContent"] h1,
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3,
[data-testid="stSidebarContent"] p  { color: #94a3b8 !important; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,212,255,0.25) !important;
}
[data-testid="stMetricLabel"]   { color: #64748b !important; font-size: 12px !important; font-weight: 500 !important; }
[data-testid="stMetricValue"]   { color: #e2e8f0 !important; font-size: 22px !important; font-weight: 700 !important; }
[data-testid="stMetricDeltaIcon"] + div { font-size: 13px !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] > div {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff 0%, #0096e0 100%) !important;
    color: #060d1b !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: .3px !important;
    padding: 10px 18px !important;
    transition: all .2s !important;
    box-shadow: 0 4px 18px rgba(0,212,255,0.3) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,212,255,0.45) !important;
}

/* ── secondary / nav button ── */
[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    transition: all .2s !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #cbd5e1 !important;
    border-color: rgba(255,255,255,0.14) !important;
}

/* ── info / success / error alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}

/* ── select box ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── number input ── */
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── slider track ── */
[data-testid="stSlider"] [role="slider"] {
    background: #00d4ff !important;
}

h1 { font-weight: 800 !important; font-size: 28px !important; letter-spacing: -.5px !important; }
h2 { font-weight: 700 !important; color: #94a3b8 !important; }
h3 { font-weight: 600 !important; color: #64748b !important; }

/* ── divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION-STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "page":              "Live Simulation",
    "sim_running":       False,
    "sim_engine":        None,
    "sim_generator":     None,
    "sim_dates":         [],
    "sim_prices":        [],
    "sim_events":        [],
    "sim_day":           0,
    "selected_agent_id": None,
    "engine":            None,          # saved engine after sim completes / pauses
    # editable simulation parameters -- now live on the Settings page
    "starting_capital":  10_000,
    "profit_goal_pct":   8,
    "loss_limit_pct":    10,
    "clone_frac":        0.5,
    "checkpoint_days":   20,
    "max_agents":        15,
    "playback_delay":    0.05,
    "days_per_frame":    3,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data/price_data.csv", parse_dates=["Date"])
    return df

try:
    price_data  = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hex_to_rgba(color_str: str, alpha: float = 0.1) -> str:
    """Convert hex/color string to Plotly-compatible rgba(...) format."""
    if not color_str:
        return f"rgba(0, 212, 255, {alpha})"
    if color_str.startswith("rgba"):
        return color_str
    if color_str.startswith("rgb"):
        return color_str.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    c = str(color_str).lstrip('#')
    if len(c) in (6, 8):
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(0, 212, 255, {alpha})"


def _clayout(title: str = "", height: int = 420, **extra) -> dict:
    """Shared dark layout dict for all Plotly figures."""
    d = dict(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0", family="Inter"), x=0, pad=dict(l=0)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.018)",
        font=dict(family="Inter", color="#64748b", size=12),
        height=height,
        margin=dict(l=0, r=0, t=52, b=0),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.06)",
            zeroline=False, tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            linecolor="rgba(255,255,255,0.06)",
            zeroline=False, tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(6,13,27,0.75)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    for k, v in extra.items():
        if k in d and isinstance(d[k], dict) and isinstance(v, dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


def _sparkline(history: list, color: str, height: int = 100) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=history, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=hex_to_rgba(color, 0.12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _pct_color(v: float) -> str:
    return "#10b981" if v >= 0 else "#ef4444"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR: Simulation parameters
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 Trading Agent Swarm")
    if data_loaded:
        st.caption("All simulation parameters now live on the **⚙️ Settings** page (top nav).")
        st.divider()
        st.markdown(f"**Starting capital:** ${st.session_state.starting_capital:,.0f}")
        st.markdown(f"**Profit goal:** +{st.session_state.profit_goal_pct}%")
        st.markdown(f"**Loss limit:** −{st.session_state.loss_limit_pct}%")
        st.markdown(f"**Max alive agents:** {st.session_state.max_agents}")
    else:
        st.error("`data/price_data.csv` not found.\nRun `python fetch_data.py` first.")

# Read current settings from session_state (edited on the Settings page)
starting_capital = st.session_state.starting_capital
profit_goal_pct  = st.session_state.profit_goal_pct
loss_limit_pct   = st.session_state.loss_limit_pct
clone_frac       = st.session_state.clone_frac
checkpoint_days  = st.session_state.checkpoint_days
max_agents       = st.session_state.max_agents
playback_delay   = st.session_state.playback_delay
days_per_frame   = st.session_state.days_per_frame

# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("🏠", "Live Simulation"),
    ("📊", "Market"),
    ("🌐", "Overview"),
    ("💰", "P&L"),
    ("🧬", "Agents"),
]


def render_nav():
    current = st.session_state.page

    # Glassmorphism header card
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,rgba(0,212,255,0.07),rgba(0,96,224,0.04));
        border: 1px solid rgba(0,212,255,0.14);
        border-radius: 20px;
        padding: 14px 24px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span style="font-size:26px;">🧬</span>
        <div>
            <div style="font-size:20px;font-weight:800;color:#e2e8f0;letter-spacing:-.5px;">
                Trading Agent Swarm
            </div>
            <div style="font-size:12px;color:#475569;margin-top:2px;">
                Self-cloning evolutionary trading simulation
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation row
    left_pad, *nav_cols, right_pad = st.columns([0.5] + [1] * len(NAV_ITEMS) + [0.5])
    for col, (icon, name) in zip(nav_cols, NAV_ITEMS):
        with col:
            is_active = (current == name) or (
                name == "Agents" and current == "Agent Detail"
            )
            if st.button(
                f"{icon} {name}" if is_active else icon,
                key=f"nav_{name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=name,
            ):
                st.session_state.page = name
                st.rerun()

    st.markdown("<hr style='margin:12px 0 24px 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def render_live_page():
    if not data_loaded:
        st.error("Run `python fetch_data.py` first to generate price data.")
        return

    # ── Controls row ────────────────────────────────────────────────────────
    ctrl_left, ctrl_right = st.columns([3, 2.5])
    with ctrl_left:
        st.markdown("## ▶ Live Simulation")

    with ctrl_left:
        gear_col, _ = st.columns([1, 6])
        with gear_col:
            with st.popover("⚙️", use_container_width=False):
                st.markdown("**Configure Simulation**")
                _cap  = st.number_input("Starting capital ($)", min_value=100,
                            value=st.session_state.starting_capital, step=500, key="_cfg_cap")
                _pg   = st.slider("Profit goal to clone (%)", 1, 50,
                            value=st.session_state.profit_goal_pct, key="_cfg_pg")
                _ll   = st.slider("Loss limit to terminate (%)", 1, 50,
                            value=st.session_state.loss_limit_pct, key="_cfg_ll")
                _cf   = st.slider("Capital fraction to clone", 0.1, 0.9,
                            value=st.session_state.clone_frac, key="_cfg_cf")
                _ckd  = st.slider("Days between checkpoints", 5, 60,
                            value=st.session_state.checkpoint_days, key="_cfg_ckd")
                _ma   = st.slider("Max alive agents", 2, 30,
                            value=st.session_state.max_agents, key="_cfg_ma")
                if st.button("💾 Save Changes", type="primary", use_container_width=True):
                    st.session_state.starting_capital = _cap
                    st.session_state.profit_goal_pct   = _pg
                    st.session_state.loss_limit_pct    = _ll
                    st.session_state.clone_frac        = _cf
                    st.session_state.checkpoint_days   = _ckd
                    st.session_state.max_agents        = _ma
                    st.success("Saved — takes effect next time you click Start.")
                    st.rerun()
        
    with ctrl_right:
        c1, c2, c3, c4 = st.columns(4)
        has_engine = st.session_state.sim_engine is not None
        is_running = st.session_state.sim_running

        start_btn = c1.button(
            "▶ Start", type="primary", use_container_width=True,
            disabled=is_running or has_engine,
        )
        pause_btn = c2.button(
            "⏸ Pause" if is_running else "▶ Resume",
            use_container_width=True,
            disabled=not has_engine,
        )
        stop_btn = c3.button(
            "⏹ Stop", use_container_width=True,
            disabled=not has_engine,
        )
        reset_btn = c4.button("↺ Reset", use_container_width=True)

    # ── Start ────────────────────────────────────────────────────────────────
    if start_btn and not st.session_state.sim_running:
        start_offset = random.randint(0, max(0, len(price_data) - checkpoint_days * 10))
        engine = EvolutionEngine(
            data=price_data,
            starting_capital=starting_capital,
            profit_goal_pct=profit_goal_pct,
            loss_limit_pct=loss_limit_pct,
            clone_capital_fraction=clone_frac,
            checkpoint_days=checkpoint_days,
            max_agents=max_agents,
            start_offset=start_offset,
        )
        st.session_state.sim_engine    = engine
        st.session_state.sim_generator = engine.run_live()
        st.session_state.sim_running   = True
        st.session_state.sim_dates     = []
        st.session_state.sim_prices    = []
        st.session_state.sim_events    = []
        st.session_state.sim_day       = 0
        st.rerun()

    # ── Pause / Resume ───────────────────────────────────────────────────────
    if pause_btn and has_engine:
        st.session_state.sim_running = not st.session_state.sim_running
        if st.session_state.sim_engine:
            st.session_state.engine = st.session_state.sim_engine
        st.rerun()

    # ── Stop ─────────────────────────────────────────────────────────────────
    if stop_btn:
        st.session_state.sim_running = False
        if st.session_state.sim_engine:
            st.session_state.engine = st.session_state.sim_engine

    # ── Reset ────────────────────────────────────────────────────────────────
    if reset_btn:
        st.session_state.sim_running   = False
        st.session_state.sim_engine    = None
        st.session_state.sim_generator = None
        st.session_state.sim_dates     = []
        st.session_state.sim_prices    = []
        st.session_state.sim_events    = []
        st.session_state.sim_day       = 0
        st.rerun()

    engine = st.session_state.sim_engine

    # ── Nothing started yet ──────────────────────────────────────────────────
    if engine is None:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        return

    # ── Status bar ───────────────────────────────────────────────────────────
    alive_n  = sum(1 for a in engine.agents.values() if a.status == "alive")
    total_n  = len(engine.agents)
    status_ph = st.empty()

    if st.session_state.sim_running:
        status_ph.info(
            f"⚡ **Running** — Day **{st.session_state.sim_day + 1:,}** | "
            f"**{alive_n}** alive / **{total_n}** total agents"
        )
    else:
        status_ph.success(
            f"⏸ **Paused** — Day **{st.session_state.sim_day + 1:,}** | "
            f"**{alive_n}** alive / **{total_n}** total agents | "
            "Switch to other pages to explore results ↑"
        )

    # ── Charts (placeholders with stable keys) ───────────────────────────────
    col_p, col_pf = st.columns(2)
    price_ph = col_p.empty()
    port_ph  = col_pf.empty()

    # ── Event log placeholder ────────────────────────────────────────────────
    event_ph = st.empty()

    # ── Render current state ─────────────────────────────────────────────────
    WINDOW = 800  # sliding window for live chart

    sim_dates  = st.session_state.sim_dates
    sim_prices = st.session_state.sim_prices
    show_dates  = sim_dates[-WINDOW:]
    show_prices = sim_prices[-WINDOW:]

    if show_dates:
        # Price chart
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=show_dates, y=show_prices,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            line=dict(color="#00d4ff", width=2),
            name="Price",
            hovertemplate="$%{y:.2f}<extra></extra>",
        ))
        fig_p.update_layout(**_clayout("📈 Live Market Price", 380))
        price_ph.plotly_chart(fig_p, use_container_width=True, key="live_price_chart")

        # Portfolio chart (all agents, colored)
        fig_pf = go.Figure()
        for a in engine.agents.values():
            if not a.portfolio_history:
                continue
            hist = a.portfolio_history[-WINDOW:]
            dh   = a.date_history[-WINDOW:]
            fig_pf.add_trace(go.Scatter(
                x=dh, y=hist,
                name=a.id,
                mode="lines",
                line=dict(
                    color=a.color,
                    width=1.5,
                    dash="dot" if a.status == "dead" else "solid",
                ),
                opacity=0.45 if a.status == "dead" else 1.0,
                hovertemplate=f"<b>{a.id}</b><br>${'{y:.2f}'}<extra></extra>",
            ))
        fig_pf.update_layout(**_clayout("💼 Agent Portfolios", 380))
        port_ph.plotly_chart(fig_pf, use_container_width=True, key="live_portfolio_chart")

    # ── Event feed ───────────────────────────────────────────────────────────
    evts = st.session_state.sim_events
    if evts:
        lines = []
        for e in reversed(evts[-12:]):
            if e["type"] == "CLONE":
                lines.append(
                    f"🧬 Day {e['day_index']:,} — **{e['parent']}** (+{e['parent_return_pct']}%) "
                    f"→ spawned **{e['child']}**"
                )
            elif e["type"] == "RESPAWN":
                lines.append(
                    f"🌱 Day {e['day_index']:,} — **{e['agent']}** auto-respawned: {e['cause']}"
                )
            else:
                lines.append(
                    f"☠️ Day {e['day_index']:,} — **{e['agent']}** terminated: {e['cause']}"
                )
        event_ph.markdown(
            "<div style='background:rgba(255,255,255,0.02);border:1px solid "
            "rgba(255,255,255,0.06);border-radius:12px;padding:14px 18px;"
            "font-size:13px;line-height:2;'>"
            + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True,
        )

    # ── Simulation loop (runs one "frame" per Streamlit rerun) ───────────────
    if st.session_state.sim_running and st.session_state.sim_generator:
        gen        = st.session_state.sim_generator
        days_done  = 0

        try:
            while days_done < days_per_frame:
                upd = next(gen)
                if upd["type"] == "day":
                    row = upd["row"]
                    st.session_state.sim_dates.append(row["Date"])
                    st.session_state.sim_prices.append(float(row["Close"]))
                    st.session_state.sim_day = upd["day_index"]
                    days_done += 1
                elif upd["type"] == "checkpoint":
                    for ev in upd["events"]:
                        st.session_state.sim_events.append(ev)
                    # Cap event list
                    if len(st.session_state.sim_events) > 200:
                        st.session_state.sim_events = st.session_state.sim_events[-200:]
        except StopIteration:
            st.session_state.sim_running = False
            st.session_state.engine      = engine

        # Cap price lists to avoid unbounded memory growth (keep 5k points)
        if len(st.session_state.sim_dates) > 5000:
            st.session_state.sim_dates  = st.session_state.sim_dates[-5000:]
            st.session_state.sim_prices = st.session_state.sim_prices[-5000:]

        if playback_delay > 0:
            time.sleep(playback_delay)

        # Save engine for other pages and trigger next frame
        st.session_state.engine = engine
        st.rerun()

    # Save engine so other pages can read it
    if engine:
        st.session_state.engine = engine


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MARKET
# ══════════════════════════════════════════════════════════════════════════════
def render_market_page():
    st.markdown("## 📊 Market Data")
    st.caption("Historical price data every agent trades against. Synthetic extension shown in purple.")

    if not data_loaded:
        st.error("`data/price_data.csv` not found — run `python fetch_data.py` first.")
        return

    # Always use raw CSV as base; overlay synthetic extension if a simulation ran
    df = price_data.copy()
    engine = st.session_state.get("engine")
    if engine is not None and len(engine.data) > len(price_data):
        df = engine.data.copy()

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    last_close = float(df["Close"].iloc[-1])
    first_close = float(df["Close"].iloc[0])
    total_ret = (last_close - first_close) / first_close * 100
    c1.metric("Latest Close", f"${last_close:,.2f}")
    c2.metric("Data points", f"{len(df):,}")
    c3.metric("Total return (full period)", f"{total_ret:+.1f}%")
    c4.metric("Date range",
              f"{str(df['Date'].iloc[0])[:10]} → {str(df['Date'].iloc[-1])[:10]}")

    # ── Price + MA chart ─────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Close", mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
        line=dict(color="#00d4ff", width=2),
    ))
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["MA20"],
            name="MA20", mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))

    # Mark synthetic region
    if "is_synthetic" in df.columns:
        synth_start = df.loc[df["is_synthetic"] == True, "Date"]
        if not synth_start.empty:
            fig.add_vrect(
                x0=synth_start.iloc[0], x1=df["Date"].iloc[-1],
                fillcolor="rgba(139,92,246,0.06)",
                line_width=1, line_color="rgba(139,92,246,0.3)",
                annotation_text="Synthetic", annotation_position="top left",
                annotation_font_color="#a78bfa",
            )

    fig.update_layout(**_clayout("Close Price + 20-day MA", 440,
                                  xaxis_title="Date", yaxis_title="Price ($)"))
    st.plotly_chart(fig, use_container_width=True, key="market_price")

    # ── Volume chart ─────────────────────────────────────────────────────────
    if "Volume" in df.columns and df["Volume"].notna().any():
        fig_v = go.Figure(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color="rgba(0,212,255,0.35)",
            name="Volume",
        ))
        fig_v.update_layout(**_clayout("Volume", 200, yaxis_title="Shares"))
        st.plotly_chart(fig_v, use_container_width=True, key="market_volume")

    # ── RSI chart ────────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        fig_r = go.Figure()
        fig_r.add_hrect(y0=0,  y1=30, fillcolor="rgba(16,185,129,0.08)", line_width=0)
        fig_r.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)",  line_width=0)
        fig_r.add_trace(go.Scatter(
            x=df["Date"], y=df["RSI"],
            name="RSI", mode="lines",
            line=dict(color="#a78bfa", width=2),
        ))
        fig_r.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1,
                        annotation_text="Overbought (70)", annotation_font_color="#ef4444")
        fig_r.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1,
                        annotation_text="Oversold (30)", annotation_font_color="#10b981")
        fig_r.update_layout(
            **_clayout("RSI (14)", 260, yaxis=dict(range=[0, 100]))
        )
        st.plotly_chart(fig_r, use_container_width=True, key="market_rsi")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def render_overview_page(engine: EvolutionEngine):
    st.markdown("## 🌐 Simulation Overview")

    all_agents   = list(engine.agents.values())
    alive_agents = [a for a in all_agents if a.status == "alive"]
    dead_agents  = [a for a in all_agents if a.status == "dead"]
    last_price   = float(engine.data["Close"].iloc[-1])
    total_value  = sum(a.portfolio_value(last_price) for a in alive_agents)
    seed_capital = sum(
        a.initial_capital for a in all_agents if a.parent_id is None
    ) or engine.starting_capital

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total agents created", len(all_agents))
    c2.metric("Currently alive", len(alive_agents))
    c3.metric("Terminated", len(dead_agents))
    c4.metric("Combined portfolio (alive)", f"${total_value:,.0f}")
    total_pnl = total_value - seed_capital
    c5.metric(
        "Net P&L",
        f"${total_pnl:+,.0f}",
        f"{total_pnl / seed_capital * 100:+.1f}%",
    )

    st.divider()

    # ── All agents portfolio chart ────────────────────────────────────────────
    fig = go.Figure()
    for a in all_agents:
        if not a.portfolio_history:
            continue
        fig.add_trace(go.Scatter(
            x=a.date_history,
            y=a.portfolio_history,
            name=a.id,
            mode="lines",
            line=dict(
                color=a.color, width=2,
                dash="dot" if a.status == "dead" else "solid",
            ),
            opacity=0.4 if a.status == "dead" else 1.0,
        ))
    fig.update_layout(**_clayout("All Agent Portfolio Values Over Time", 480,
                                  xaxis_title="Date", yaxis_title="Portfolio ($)"))
    st.plotly_chart(fig, use_container_width=True, key="overview_portfolios")

    # ── Event timeline ────────────────────────────────────────────────────────
    st.markdown("### 🗓 Event Timeline")
    if not engine.events:
        st.info("No clone or kill events yet. Run the simulation longer or loosen the thresholds.")
    else:
        for e in engine.events:
            if e["type"] == "CLONE":
                st.success(
                    f"Day {e['day_index']:,} 🧬 **{e['parent']}** (+{e['parent_return_pct']}%) "
                    f"→ spawned **{e['child']}**"
                )
            else:
                st.error(
                    f"Day {e['day_index']:,} ☠️ **{e['agent']}** terminated — {e['cause']}"
                )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: P&L
# ══════════════════════════════════════════════════════════════════════════════
def render_pnl_page(engine: EvolutionEngine):
    st.markdown("## 💰 Profit & Loss")

    all_agents = list(engine.agents.values())
    last_price  = float(engine.data["Close"].iloc[-1])

    rows = []
    for a in all_agents:
        cur_val = a.portfolio_value(last_price)
        pnl     = cur_val - a.initial_capital
        pnl_pct = a.return_pct(last_price)
        rows.append({
            "Agent":             a.id,
            "Gen":               a.generation,
            "Status":            "🟢 Alive" if a.status == "alive" else "🔴 Dead",
            "Starting ($)":      round(a.initial_capital, 2),
            "Current Value ($)": round(cur_val, 2),
            "P&L ($)":           round(pnl, 2),
            "Return (%)":        round(pnl_pct, 2),
            "Parent":            a.parent_id or "— seed",
        })

    df_pnl = pd.DataFrame(rows)

    # ── Summary metrics ──────────────────────────────────────────────────────
    seed_capital = sum(
        a.initial_capital for a in all_agents if a.parent_id is None
    ) or engine.starting_capital
    total_cur   = sum(
        a.portfolio_value(last_price) for a in all_agents if a.status == "alive"
    )
    total_pnl   = total_cur - seed_capital
    total_invested = sum(a.initial_capital for a in all_agents)

    best  = max(all_agents, key=lambda a: a.return_pct(last_price))
    worst = min(all_agents, key=lambda a: a.return_pct(last_price))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Deployed Capital", f"${seed_capital:,.0f}")
    c2.metric("Total Capital Allocated", f"${total_invested:,.0f}")
    c3.metric("Alive Portfolio Value",   f"${total_cur:,.0f}")
    c4.metric(
        "Net P&L",
        f"${total_pnl:+,.0f}",
        f"{total_pnl / seed_capital * 100:+.1f}%",
    )
    c5.metric(
        "Best Agent",
        best.id,
        f"{best.return_pct(last_price):+.1f}%",
    )

    st.divider()

    # ── P&L bar chart ─────────────────────────────────────────────────────────
    colors = [
        ("#10b981" if r["P&L ($)"] >= 0 else "#ef4444") for r in rows
    ]
    fig_bar = go.Figure(go.Bar(
        x=df_pnl["Agent"],
        y=df_pnl["P&L ($)"],
        marker_color=colors,
        text=[f"${v:+,.0f}" for v in df_pnl["P&L ($)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_bar.update_layout(
        **_clayout("Agent P&L ($)", 360,
                   yaxis_title="Profit / Loss ($)", xaxis_title="Agent")
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="pnl_bar")

    # ── Return % scatter ──────────────────────────────────────────────────────
    fig_ret = go.Figure(go.Bar(
        x=df_pnl["Agent"],
        y=df_pnl["Return (%)"],
        marker_color=[("#10b981" if r >= 0 else "#ef4444") for r in df_pnl["Return (%)"]],
        text=[f"{v:+.1f}%" for v in df_pnl["Return (%)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig_ret.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig_ret.update_layout(
        **_clayout("Agent Return (%)", 300,
                   yaxis_title="Return (%)", xaxis_title="Agent")
    )
    st.plotly_chart(fig_ret, use_container_width=True, key="pnl_return_bar")

    st.divider()

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown("### Per-Agent Breakdown")
    st.dataframe(
        df_pnl.sort_values("Return (%)", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AGENTS (cards only – click to navigate to detail)
# ══════════════════════════════════════════════════════════════════════════════
def render_agents_page(engine: EvolutionEngine):
    st.markdown("## 🧬 Agents")
    st.caption(
        "Click an agent card to see its full detail, thinking log, and genome. "
        "Dashed sparklines = terminated agents."
    )

    all_agents = list(engine.agents.values())
    last_price  = float(engine.data["Close"].iloc[-1])

    by_gen: dict[int, list] = {}
    for a in all_agents:
        by_gen.setdefault(a.generation, []).append(a)

    for gen in sorted(by_gen):
        agents_in_gen = by_gen[gen]
        gen_label = "🌱 Seed" if gen == 0 else f"Generation {gen}"
        st.markdown(
            f"<div style='color:#64748b;font-size:13px;font-weight:600;"
            f"letter-spacing:.5px;text-transform:uppercase;margin:16px 0 10px'>"
            f"{gen_label} — {len(agents_in_gen)} agent(s)</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(min(len(agents_in_gen), 5))
        for col, a in zip(cols, agents_in_gen):
            with col:
                ret = a.return_pct(last_price)
                is_alive = a.status == "alive"
                status_dot = "🟢" if is_alive else "🔴"
                ret_color  = _pct_color(ret)

                # Card HTML (visual only)
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.025);
                        border: 1px solid {a.color}40;
                        border-top: 3px solid {a.color};
                        border-radius: 14px;
                        padding: 12px 14px 6px;
                        margin-bottom: 4px;
                    ">
                        <div style="font-size:12px;color:{a.color};font-weight:700;
                                    letter-spacing:.3px;">{status_dot} {a.id}</div>
                        <div style="font-size:11px;color:#475569;margin:2px 0 6px;">
                            Gen {a.generation} · {a.parent_id or 'seed'}
                        </div>
                        <div style="font-size:18px;font-weight:800;
                                    color:{ret_color};">{ret:+.1f}%</div>
                        <div style="font-size:11px;color:#475569;">return</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Sparkline
                if a.portfolio_history:
                    sp_fig = _sparkline(a.portfolio_history[-200:], a.color, height=80)
                    st.plotly_chart(sp_fig, use_container_width=True, key=f"spark_{a.id}")

                # Click button → navigate to detail
                if st.button(
                    f"View {a.id}",
                    key=f"detail_btn_{a.id}",
                    use_container_width=True,
                ):
                    st.session_state.selected_agent_id = a.id
                    st.session_state.page = "Agent Detail"
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AGENT DETAIL
# ══════════════════════════════════════════════════════════════════════════════
def render_agent_detail_page(engine: EvolutionEngine):
    # ── Back button ───────────────────────────────────────────────────────────
    if st.button("← Back to Agents", type="secondary"):
        st.session_state.page = "Agents"
        st.session_state.selected_agent_id = None
        st.rerun()

    agent_id = st.session_state.get("selected_agent_id")
    if not agent_id or agent_id not in engine.agents:
        st.warning("No agent selected. Go back to Agents and click a card.")
        return

    agent      = engine.agents[agent_id]
    last_price = float(engine.data["Close"].iloc[-1])
    ret        = agent.return_pct(last_price)
    cur_val    = agent.portfolio_value(last_price)
    pnl        = cur_val - agent.initial_capital
    is_alive   = agent.status == "alive"

    # ── Agent header ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg,{agent.color}12,{agent.color}06);
            border: 1px solid {agent.color}35;
            border-left: 4px solid {agent.color};
            border-radius: 16px;
            padding: 18px 24px;
            margin-bottom: 20px;
            display: flex; align-items: center; gap: 20px;
        ">
            <div style="font-size:36px;">{'🟢' if is_alive else '🔴'}</div>
            <div>
                <div style="font-size:22px;font-weight:800;color:#e2e8f0;">{agent.id}</div>
                <div style="font-size:13px;color:#64748b;margin-top:4px;">
                    Generation {agent.generation} · Parent: {agent.parent_id or '— seed'} ·
                    Status: <span style="color:{agent.color}">{agent.status.upper()}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_alive and agent.cause_of_death:
        st.error(f"☠️ Cause of termination: {agent.cause_of_death}")

    # ── Metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Capital",  f"${agent.initial_capital:,.2f}")
    c2.metric("Current Value",     f"${cur_val:,.2f}")
    c3.metric("Net P&L",           f"${pnl:+,.2f}", f"{ret:+.1f}%")
    c4.metric("Shares Held",       f"{agent.position:.4f}")

    st.divider()

    # ── Portfolio chart ───────────────────────────────────────────────────────
    if agent.portfolio_history:
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(
            x=agent.date_history,
            y=agent.portfolio_history,
            mode="lines",
            fill="tozeroy",
            fillcolor=hex_to_rgba(agent.color, 0.12),
            line=dict(color=agent.color, width=2.5),
            name="Portfolio Value",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ))
        # Starting capital reference line
        fig_pf.add_hline(
            y=agent.initial_capital,
            line_dash="dash", line_color="rgba(255,255,255,0.25)",
            line_width=1,
            annotation_text=f"Start ${agent.initial_capital:,.0f}",
            annotation_font_color="#64748b",
        )
        fig_pf.update_layout(
            **_clayout(f"{agent.id} — Portfolio Value Over Time", 380,
                       xaxis_title="Date", yaxis_title="Value ($)")
        )
        st.plotly_chart(fig_pf, use_container_width=True, key=f"detail_portfolio_{agent_id}")

    st.divider()

    # ── Genome comparison ──────────────────────────────────────────────────────
    st.markdown("### 🧬 Genome (Strategy Parameters)")
    genome_rows = []
    parent_genome = (
        engine.agents[agent.parent_id].genome
        if agent.parent_id and agent.parent_id in engine.agents
        else None
    )
    for k, v in agent.genome.items():
        parent_v = parent_genome.get(k) if parent_genome else None
        mutated  = "⬅ mutated" if parent_v is not None and parent_v != v else ""
        genome_rows.append({
            "Parameter":    k,
            "Value":        v,
            "Parent Value": parent_v if parent_v is not None else "— (seed)",
            "":             mutated,
        })
    st.dataframe(
        pd.DataFrame(genome_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Thinking / Decision log ───────────────────────────────────────────────
    st.markdown("### 🧠 Thinking Process (Decision Log)")
    st.caption("Last 500 decisions — BUY 🟢 · SELL 🔴 · HOLD ⬜")

    if agent.decision_log:
        log_df = pd.DataFrame(agent.decision_log[-500:])
        log_df = log_df[["date", "price", "rsi", "action", "reason", "portfolio_value"]].copy()
        log_df.columns = ["Date", "Price ($)", "RSI", "Action", "Reason", "Portfolio ($)"]
        # Prefix action with an emoji instead of using pandas Styler (Styler + hide_index
        # breaks on some pandas/Streamlit version combinations)
        action_emoji = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⬜ HOLD"}
        log_df["Action"] = log_df["Action"].map(lambda a: action_emoji.get(a, a))
        st.dataframe(log_df, use_container_width=True, height=480, hide_index=True)
    else:
        st.info("No decisions recorded yet.")


def render_settings_page():
    st.markdown("## ⚙️ Simulation Settings & Parameters")
    st.caption("Adjust parameters here, then go to **🏠 Live Simulation** and click **▶ Start**.")

    if st.session_state.sim_running or st.session_state.sim_engine is not None:
        st.warning(
            "A simulation is currently running or paused. Changes here won't apply until you "
            "**↺ Reset** the simulation from the Live Simulation page."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 💵 Capital & Risk Parameters")
        st.number_input(
            "Starting capital ($)", min_value=100, max_value=10_000_000,
            step=500, key="starting_capital",
        )
        st.slider("Profit goal to clone (%)", 1, 50, key="profit_goal_pct")
        st.slider("Loss limit to terminate (%)", 1, 50, key="loss_limit_pct")
        st.slider("Capital fraction given to clone", 0.1, 0.9, step=0.05, key="clone_frac")

    with c2:
        st.markdown("### ⏱ Execution & Population")
        st.slider("Days between checkpoints", 5, 60, key="checkpoint_days")
        st.slider("Max alive agents", 2, 30, key="max_agents")
        st.slider("Playback frame delay (seconds)", 0.0, 2.0, step=0.01, key="playback_delay")
        st.slider("Days simulated per frame", 1, 30, key="days_per_frame")

    st.divider()
    st.markdown("### Current configuration")
    st.json({
        "starting_capital": st.session_state.starting_capital,
        "profit_goal_pct": st.session_state.profit_goal_pct,
        "loss_limit_pct": st.session_state.loss_limit_pct,
        "clone_capital_fraction": st.session_state.clone_frac,
        "checkpoint_days": st.session_state.checkpoint_days,
        "max_agents": st.session_state.max_agents,
        "playback_delay_sec": st.session_state.playback_delay,
        "days_per_frame": st.session_state.days_per_frame,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
render_nav()

page    = st.session_state.page
engine  = st.session_state.get("engine")

if page == "Live Simulation":
    render_live_page()

elif not data_loaded:
    st.error("Run `python fetch_data.py` first to fetch market data.")

elif page == "Market":
    render_market_page()

elif page == "Settings":
    render_settings_page()

elif engine is None:
    st.info(
        "🚀 Go to **🏠 Live Simulation** and click **▶ Start** first — "
        "then come back to explore results on this page."
    )

elif page == "Overview":
    render_overview_page(engine)

elif page == "P&L":
    render_pnl_page(engine)

elif page == "Agents":
    render_agents_page(engine)

elif page == "Agent Detail":
    render_agent_detail_page(engine)
