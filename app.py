import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="Greek Exposures Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown('''
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00FFFF; font-family: monospace; }
    .stButton > button { font-family: monospace; }
    .stSuccess { background-color: #1a2332 !important; }
    .stPlotlyChart {
        display: flex;
        justify-content: center;
    }
</style>
''', unsafe_allow_html=True)


# ============================================================
# FONCTIONS BLACK-SCHOLES
# ============================================================
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def black_scholes_theta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365.0

def black_scholes_vanna(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return -norm.pdf(d1) * d2 / sigma / 100.0

def black_scholes_charm(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    phi_d1 = norm.pdf(d1)
    charm_call = -phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    if option_type == 'call':
        charm = charm_call
    else:
        charm = charm_call - r * np.exp(-r * T)
    return charm / 365.0


# ============================================================
# FONCTIONS DE CALCUL DES EXPOSURES
# ============================================================
def calculate_gex_chain(chain_calls, chain_puts, spot, expiration_date, risk_free_rate=0.045):
    now = datetime.now()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - now).days / 365.0, 1 / 365.0)
    records = []
    for df, sign in [(chain_calls, 1), (chain_puts, -1)]:
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            strike, oi, iv = row.get("strike", 0), row.get("openInterest", 0), row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            gamma = black_scholes_gamma(spot, strike, T, risk_free_rate, iv)
            records.append({"strike": strike, "value": sign * oi * gamma * (spot ** 2) * 100})
    return pd.DataFrame(records)

def calculate_tex_chain(chain_calls, chain_puts, spot, expiration_date, risk_free_rate=0.045):
    now = datetime.now()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - now).days / 365.0, 1 / 365.0)
    records = []
    for df, option_type in [(chain_calls, "call"), (chain_puts, "put")]:
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            strike, oi, iv = row.get("strike", 0), row.get("openInterest", 0), row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            theta = black_scholes_theta(spot, strike, T, risk_free_rate, iv, option_type)
            records.append({"strike": strike, "value": -1 * oi * theta * spot * 100})
    return pd.DataFrame(records)

def calculate_vanna_exposure_chain(chain_calls, chain_puts, spot, expiration_date, risk_free_rate=0.045):
    now = datetime.now()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - now).days / 365.0, 1 / 365.0)
    records = []
    for df, sign in [(chain_calls, -1), (chain_puts, 1)]:
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            strike, oi, iv = row.get("strike", 0), row.get("openInterest", 0), row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            vanna = black_scholes_vanna(spot, strike, T, risk_free_rate, iv)
            records.append({"strike": strike, "value": sign * oi * vanna * spot * 100})
    return pd.DataFrame(records)

def calculate_charm_exposure_chain(chain_calls, chain_puts, spot, expiration_date, risk_free_rate=0.045):
    now = datetime.now()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - now).days / 365.0, 1 / 365.0)
    records = []
    for df, option_type in [(chain_calls, "call"), (chain_puts, "put")]:
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            strike, oi, iv = row.get("strike", 0), row.get("openInterest", 0), row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            charm = black_scholes_charm(spot, strike, T, risk_free_rate, iv, option_type)
            records.append({"strike": strike, "value": -1 * oi * charm * spot * 100})
    return pd.DataFrame(records)


# ============================================================
# CONSTRUCTION MATRICE + HEATMAP
# ============================================================
def build_matrix(data_dict, grid_strikes, strike_step=2.5):
    exp_labels = list(data_dict.keys())
    matrix, text_annotations = [], []
    for strike in grid_strikes:
        row_values, row_text = [], []
        for exp in exp_labels:
            series = data_dict.get(exp, pd.Series(dtype=float))
            if strike in series.index:
                val = series[strike]
            else:
                diffs = (series.index - strike).map(abs)
                val = series.iloc[diffs.argmin()] if len(diffs) > 0 and diffs.min() <= strike_step * 0.6 else 0.0
            row_values.append(val)
            # Formatage texte
            if val == 0:
                row_text.append("$0")
            elif abs(val) >= 1e9:
                row_text.append(f"${val/1e9:.2f}B")
            elif abs(val) >= 1e6:
                row_text.append(f"${val/1e6:.2f}M")
            elif abs(val) >= 1e3:
                row_text.append(f"${val/1e3:.1f}K")
            else:
                row_text.append(f"${val:.0f}")
        matrix.append(row_values)
        text_annotations.append(row_text)
    return np.array(matrix, dtype=float), np.array(text_annotations), exp_labels


def create_single_heatmap(z, text, labels, grid_strikes, spot, title, colorbar_title):
    """Crée une figure Plotly pour UNE heatmap avec le spot marker."""
    colorscale = [
        [0.0, "#211340"], [0.2, "#EA00FF"], [0.35, "#FF00DD"],
        [0.5, "#D1C1AE"], [0.65, "#3552FC"], [0.8, "#101A6B"], [1.0, "#192040"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=labels, y=[float(s) for s in grid_strikes],
            text=text, texttemplate="%{text}",
            textfont={"size": 11, "color": "white", "family": "monospace"},
            colorscale=colorscale, zmid=0, showscale=True,
            colorbar=dict(
                title=dict(text=f"<b>{colorbar_title}</b>", font=dict(color="white", size=12, family="monospace")),
                tickfont=dict(color="white", size=9, family="monospace"),
                len=0.75, thickness=22, x=1.02
            ),
            hovertemplate="Strike %{y} | Exp %{x} | Val %{z:+,.0f}<extra></extra>"
        )
    )

    # Label spot à gauche
    spot_strike = min(grid_strikes, key=lambda s: abs(s - spot))
    fig.add_annotation(
        x=-0.5, y=spot_strike,
        text=f"<b>▶ SPOT {spot:,.0f}</b>",
        showarrow=False,
        font=dict(color="#00FFFF", size=10, family="monospace"),
        xref="x", yref="y", xanchor="right", yanchor="middle",
        bgcolor="rgba(0,0,0,0.85)", bordercolor="#00FFFF", borderwidth=1, borderpad=4
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#aaa'>{datetime.now().strftime('%Y-%m-%d %H:%M')} — Spot: ${spot:,.2f}</span>",
            x=0.5, xanchor="center", font=dict(size=18, color="white", family="monospace")
        ),
        xaxis=dict(
            title=dict(text="Expiration Date", font=dict(color="#666", size=11, family="monospace")),
            tickfont=dict(color="white", size=10, family="monospace"),
            showgrid=False, showline=True, linecolor="#333", type="category"
        ),
        yaxis=dict(
            title=dict(text="Strike", font=dict(color="#666", size=11, family="monospace")),
            tickfont=dict(color="white", size=9, family="monospace"),
            showgrid=False, showline=True, linecolor="#333", dtick=5
        ),
        paper_bgcolor="#000", plot_bgcolor="#000",
        width=794, height=1123,
        margin=dict(l=70, r=100, t=80, b=40)
    )
    return fig


def create_all_heatmaps(all_data, grid_strikes, spot, symbol):
    """Crée la figure avec les 4 heatmaps empilées verticalement."""
    colorscale = [
        [0.0, "#211340"], [0.2, "#EA00FF"], [0.35, "#FF00DD"],
        [0.5, "#D1C1AE"], [0.65, "#3552FC"], [0.8, "#101A6B"], [1.0, "#192040"]
    ]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(f'{symbol} GEX', f'{symbol} THETA', f'{symbol} VANNA', f'{symbol} CHARM'),
        vertical_spacing=0.04,
        specs=[[{"type": "heatmap"}]] * 4
    )

    colorbar_positions = [0.89, 0.63, 0.37, 0.11]
    colorbar_titles = ["GEX", "TEX", "Vanna", "CEX"]

    for i, (key, cb_title, cb_y) in enumerate(zip(
        ["gex", "theta", "vanna", "charm"], colorbar_titles, colorbar_positions), start=1
    ):
        z, text, labels = all_data[key]
        fig.add_trace(
            go.Heatmap(
                z=z, x=labels, y=[float(s) for s in grid_strikes],
                text=text, texttemplate="%{text}",
                textfont={"size": 9, "color": "white", "family": "monospace"},
                colorscale=colorscale, zmid=0, showscale=True,
                colorbar=dict(
                    title=dict(text=f"<b>{cb_title}</b>", font=dict(color="white", size=10, family="monospace")),
                    tickfont=dict(color="white", size=8, family="monospace"),
                    len=0.22, thickness=18, x=1.02, y=cb_y
                ),
                hovertemplate=f"Strike %{{y}} | Exp %{{x}} | {cb_title} %{{z:+,.0f}}<extra></extra>"
            ),
            row=i, col=1
        )

        # Spot marker sur chaque subplot
        spot_strike = min(grid_strikes, key=lambda s: abs(s - spot))
        fig.add_annotation(
            x=-0.5, y=spot_strike,
            text=f"<b>▶ {spot:,.0f}</b>",
            showarrow=False,
            font=dict(color="#00FFFF", size=8, family="monospace"),
            xref=f"x{i}" if i > 1 else "x",
            yref=f"y{i}" if i > 1 else "y",
            xanchor="right", yanchor="middle",
            bgcolor="rgba(0,0,0,0.85)", bordercolor="#00FFFF", borderwidth=1, borderpad=3
        )

    # Axes
    for i in range(1, 5):
        fig.update_xaxes(
            title=dict(text="Expiration", font=dict(color="#666", size=10, family="monospace")),
            tickfont=dict(color="white", size=9, family="monospace"),
            showgrid=False, showline=True, linecolor="#333", type="category", row=i, col=1
        )
        fig.update_yaxes(
            title=dict(text="Strike", font=dict(color="#666", size=10, family="monospace")),
            tickfont=dict(color="white", size=7, family="monospace"),
            showgrid=False, showline=True, linecolor="#333", dtick=5, row=i, col=1
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{symbol} — GREEK EXPOSURES DASHBOARD</b><br>"
                f"<span style='font-size:12px;color:#aaa'>{datetime.now().strftime('%Y-%m-%d %H:%M')} — Spot: ${spot:,.2f}</span>"
            ),
            x=0.5, xanchor="center", font=dict(size=16, color="white", family="monospace")
        ),
        paper_bgcolor="#000", plot_bgcolor="#000",
        height=4800, width=794, showlegend=False,
        margin=dict(l=60, r=110, t=100, b=40)
    )
    return fig


# ============================================================
# SECTION GAMMA — calculs + graphique
# ============================================================
def compute_gamma_profile(chain_calls, chain_puts, spot, expiration_date, risk_free_rate=0.045):
    """
    Retourne un dict avec tout ce qu'on a besoin pour le mode Gamma :
      - gex_series   : Series strike → net GEX
      - call_gex     : Series strike → GEX calls seuls
      - put_gex      : Series strike → GEX puts seuls (négatif)
      - net_gex_oi   : somme totale net GEX
      - vol_trigger  : strike où le GEX cumulé (depuis le spot vers le bas) croise zéro
      - call_wall    : strike du max GEX calls
      - put_wall     : strike du max |GEX puts|
      - major_wall   : strike du max |net GEX|
    """
    now = datetime.now()
    exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
    T = max((exp_dt - now).days / 365.0, 1 / 365.0)

    call_records, put_records = [], []

    # --- Calls ---
    if chain_calls is not None and len(chain_calls) > 0:
        for _, row in chain_calls.iterrows():
            strike = row.get("strike", 0)
            oi     = row.get("openInterest", 0)
            iv     = row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            gamma = black_scholes_gamma(spot, strike, T, risk_free_rate, iv)
            call_records.append({"strike": strike, "value": oi * gamma * (spot ** 2) * 100})

    # --- Puts (négatif) ---
    if chain_puts is not None and len(chain_puts) > 0:
        for _, row in chain_puts.iterrows():
            strike = row.get("strike", 0)
            oi     = row.get("openInterest", 0)
            iv     = row.get("impliedVolatility", 0)
            if pd.isna(oi) or oi < 1 or pd.isna(iv) or iv <= 0 or iv > 5 or strike <= 0:
                continue
            gamma = black_scholes_gamma(spot, strike, T, risk_free_rate, iv)
            put_records.append({"strike": strike, "value": -1 * oi * gamma * (spot ** 2) * 100})

    call_gex = pd.DataFrame(call_records).groupby("strike")["value"].sum() if call_records else pd.Series(dtype=float)
    put_gex  = pd.DataFrame(put_records).groupby("strike")["value"].sum()  if put_records  else pd.Series(dtype=float)

    # Net GEX par strike
    all_strikes = sorted(set(call_gex.index) | set(put_gex.index))
    net = pd.Series(
        {s: call_gex.get(s, 0.0) + put_gex.get(s, 0.0) for s in all_strikes}
    )

    # --- Vol Trigger : on cumule le net GEX depuis le spot vers le bas ---
    below_spot = net[net.index <= spot].sort_index(ascending=False)  # du spot vers le bas
    cumsum     = below_spot.cumsum()
    # Le vol trigger est le premier strike où le cumul devient négatif
    neg_mask   = cumsum[cumsum < 0]
    vol_trigger = neg_mask.index[0] if len(neg_mask) > 0 else below_spot.index[-1] if len(below_spot) > 0 else spot

    # --- Walls ---
    call_wall   = call_gex.idxmax()  if len(call_gex) > 0  else spot
    put_wall    = put_gex.idxmin()   if len(put_gex)  > 0  else spot   # idxmin car les puts sont négatifs
    major_wall  = net.abs().idxmax() if len(net) > 0        else spot

    # Net GEX total
    net_gex_oi = float(net.sum())

    return {
        "net":            net,
        "call_gex":       call_gex,
        "put_gex":        put_gex,
        "net_gex_oi":     net_gex_oi,
        "vol_trigger":    vol_trigger,
        "call_wall":      call_wall,
        "put_wall":       put_wall,
        "major_wall":     major_wall,
    }


def create_gamma_chart(profile, spot, symbol, expiration_date):
    """Graphique horizontal bars net GEX + lignes de référence."""
    net       = profile["net"]
    strikes   = net.index.tolist()
    values    = net.values.tolist()

    vol_trigger = profile["vol_trigger"]
    call_wall   = profile["call_wall"]
    put_wall    = profile["put_wall"]
    major_wall  = profile["major_wall"]

    # Couleurs : vert si positif, rouge si négatif
    colors = ["#00cc66" if v >= 0 else "#cc3333" for v in values]

    fig = go.Figure()

    # --- Bars horizontales (x = valeur GEX, y = strike) ---
    fig.add_trace(go.Bar(
        x=values,
        y=strikes,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        hovertemplate="Strike: $%{y:,.0f}<br>Net GEX: %{x:+,.0f}<extra></extra>",
        showlegend=False,
    ))

    # --- Lignes horizontales de référence ---
    # Chaque tuple : (strike, couleur, dash, label, position du label)
    lines = [
        (spot,          "#00FFFF", "solid",   f"SPOT  ${spot:,.0f}",                "right"),
        (vol_trigger,   "#FFD700", "dashdot", f"VOL TRIGGER  ${vol_trigger:,.0f}",  "right"),
        (major_wall,    "#FF6600", "dash",    f"MAJOR WALL  ${major_wall:,.0f}",    "right"),
        (call_wall,     "#4488FF", "dot",     f"CALL WALL  ${call_wall:,.0f}",      "right"),
        (put_wall,      "#FF4488", "dot",     f"PUT WALL  ${put_wall:,.0f}",        "left"),
    ]

    for strike_lvl, color, dash, label, ann_pos in lines:
        fig.add_hline(
            y=strike_lvl,
            line=dict(color=color, width=1.2, dash=dash),
            annotation=dict(
                text=f"<b>{label}</b>",
                font=dict(color=color, size=11, family="monospace"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=color, borderwidth=1,
            ),
            annotation_position=ann_pos,
        )

    # --- Layout ---
    exp_dt  = datetime.strptime(expiration_date, "%Y-%m-%d")
    exp_fmt = exp_dt.strftime("%b %d '%y")

    fig.update_layout(
        title=dict(
            text=f"<b>${symbol} NET GEX (OPEN INTEREST) for {exp_fmt}</b>",
            x=0.5, xanchor="center",
            font=dict(size=15, color="white", family="monospace")
        ),
        xaxis=dict(
            title=dict(text="GEX (OPEN INTEREST)", font=dict(color="#666", size=11, family="monospace")),
            tickfont=dict(color="#aaa", size=9, family="monospace"),
            showgrid=True, gridcolor="#222", gridwidth=1,
            showline=True, linecolor="#444",
            tickformat="~s",           # 10M, 20M …
            zeroline=True, zerolinecolor="#444", zerolinewidth=1,
        ),
        yaxis=dict(
            title=dict(text="STRIKE", font=dict(color="#666", size=11, family="monospace")),
            tickfont=dict(color="#aaa", size=9.5, family="monospace"),
            showgrid=False,
            showline=True, linecolor="#333",
            tickprefix="$", tickformat=",.0f",
            dtick=5,
        ),
        paper_bgcolor="#000", plot_bgcolor="#0a0a0a",
        width=None, height=950,
        margin=dict(l=75, r=155, t=50, b=55),
        bargap=0.15,
    )
    return fig


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Greek Exposures Dashboard")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    symbol = st.text_input("Ticker Symbol", value="^SPX")
    n_expirations = st.slider("Nombre d'expirations", 1, 10, 5)
    strike_range = st.slider("Range strikes (±pts)", 50, 300, 100, 10)
    risk_free_rate = 0.045  # Taux sans risque fixe à 4.5%

    st.markdown("---")
    st.header("📈 Mode")
    mode = st.radio("Choisir un mode", ["Heatmaps", "Skew", "Gamma"], index=0)

    if mode == "Heatmaps":
        st.markdown("---")
        greek_options = ["Tous", "GEX", "Theta", "Vanna", "Charm"]
        selected_greek = st.radio("Choisir un Greek", greek_options, index=0)
        skew_n_exp = 5
        skew_strike_range = 300
    elif mode == "Skew":
        selected_greek = None
        st.markdown("---")
        skew_n_exp = st.slider("Nombre d'expirations (Skew)", 1, 10, 5)
        skew_strike_range = st.slider("Range strikes Skew (±pts)", 100, 600, 300, 50)
    else:  # Gamma
        selected_greek = None
        skew_n_exp = 5
        skew_strike_range = 300
        st.markdown("---")
        gamma_exp_index = st.slider("Expiration Gamma (index)", 0, 10, 0)

    st.markdown("---")
    calculate_btn = st.button("🚀 Calculer", type="primary", use_container_width=True)


# --- MAIN ---
if calculate_btn:
    with st.spinner(f"📡 Récupération des données pour {symbol}..."):
        try:
            ticker = yf.Ticker(symbol)
            spot = ticker.history(period="2d")["Close"].iloc[-1]

            # Info bar
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Spot", f"${spot:,.2f}")

            # ============================================================
            # MODE GREEKS
            # ============================================================
            if mode == "Heatmaps":
                col2.metric("📅 Expirations", f"{n_expirations}")
                col3.metric("📐 Range", f"±{strike_range} pts")

                expirations = ticker.options[:n_expirations]
                STRIKE_STEP = 2.5
                strike_min = spot - strike_range
                strike_max = spot + strike_range

                gex_data, theta_data, vanna_data, charm_data = {}, {}, {}, {}

                progress = st.progress(0, text="Fetching option chains...")
                total = len(expirations)

                for idx, exp in enumerate(expirations):
                    chain = ticker.option_chain(exp)

                    df = calculate_gex_chain(chain.calls, chain.puts, spot, exp, risk_free_rate)
                    if len(df) > 0:
                        gex_data[exp] = df.groupby("strike")["value"].sum()

                    df = calculate_tex_chain(chain.calls, chain.puts, spot, exp, risk_free_rate)
                    if len(df) > 0:
                        theta_data[exp] = df.groupby("strike")["value"].sum()

                    df = calculate_vanna_exposure_chain(chain.calls, chain.puts, spot, exp, risk_free_rate)
                    if len(df) > 0:
                        vanna_data[exp] = df.groupby("strike")["value"].sum()

                    df = calculate_charm_exposure_chain(chain.calls, chain.puts, spot, exp, risk_free_rate)
                    if len(df) > 0:
                        charm_data[exp] = df.groupby("strike")["value"].sum()

                    progress.progress((idx + 1) / total, text=f"Fetching... {idx+1}/{total} expirations")

                progress.empty()

                all_strikes = set()
                for d in [gex_data, theta_data, vanna_data, charm_data]:
                    for s in d.values():
                        all_strikes.update(s.index)

                grid_strikes = sorted(set(
                    round(s / STRIKE_STEP) * STRIKE_STEP
                    for s in all_strikes if strike_min <= s <= strike_max
                ))

                gex_z, gex_text, gex_labels = build_matrix(gex_data, grid_strikes)
                theta_z, theta_text, theta_labels = build_matrix(theta_data, grid_strikes)
                vanna_z, vanna_text, vanna_labels = build_matrix(vanna_data, grid_strikes)
                charm_z, charm_text, charm_labels = build_matrix(charm_data, grid_strikes)

                if selected_greek == "Tous":
                    all_data = {
                        "gex": (gex_z, gex_text, gex_labels),
                        "theta": (theta_z, theta_text, theta_labels),
                        "vanna": (vanna_z, vanna_text, vanna_labels),
                        "charm": (charm_z, charm_text, charm_labels),
                    }
                    fig = create_all_heatmaps(all_data, grid_strikes, spot, symbol)
                    st.plotly_chart(fig, use_container_width=False)
                else:
                    mapping = {
                        "GEX":   (gex_z, gex_text, gex_labels, f"{symbol} — Gamma Exposure (GEX)", "GEX"),
                        "Theta": (theta_z, theta_text, theta_labels, f"{symbol} — Theta Exposure (TEX)", "TEX"),
                        "Vanna": (vanna_z, vanna_text, vanna_labels, f"{symbol} — Vanna Exposure", "Vanna"),
                        "Charm": (charm_z, charm_text, charm_labels, f"{symbol} — Charm Exposure (CEX)", "CEX"),
                    }
                    z, text, labels, title, cb_title = mapping[selected_greek]
                    fig = create_single_heatmap(z, text, labels, grid_strikes, spot, title, cb_title)
                    st.plotly_chart(fig, use_container_width=False)

            # ============================================================
            # MODE SKEW
            # ============================================================
            elif mode == "Skew":
                col2.metric("📅 Expirations", f"{skew_n_exp}")
                col3.metric("📐 Range Skew", f"±{skew_strike_range} pts")

                expirations = ticker.options[:skew_n_exp]
                strike_min_skew = spot - skew_strike_range
                strike_max_skew = spot + skew_strike_range

                # Palette de couleurs pour chaque expiration (style arc-en-ciel comme l'image)
                colors = [
                    "#FF0000", "#FF4500", "#FF8C00", "#FFA500", "#FFD700",
                    "#ADFF2F", "#00FF00", "#00CED1", "#1E90FF", "#8A2BE2"
                ]

                fig = go.Figure()

                progress = st.progress(0, text="Fetching skew data...")
                total = len(expirations)

                for idx, exp in enumerate(expirations):
                    chain = ticker.option_chain(exp)

                    # On merge calls + puts pour avoir la IV sur tous les strikes
                    calls = chain.calls[["strike", "impliedVolatility"]].copy()
                    puts = chain.puts[["strike", "impliedVolatility"]].copy()

                    # Pour chaque strike, on prend le mid entre call IV et put IV
                    merged = pd.merge(
                        calls.rename(columns={"impliedVolatility": "iv_call"}),
                        puts.rename(columns={"impliedVolatility": "iv_put"}),
                        on="strike", how="outer"
                    )
                    merged["iv_mid"] = merged[["iv_call", "iv_put"]].mean(axis=1)

                    # Filtres
                    merged = merged.dropna(subset=["iv_mid"])
                    merged = merged[(merged["strike"] >= strike_min_skew) & (merged["strike"] <= strike_max_skew)]
                    merged = merged[merged["iv_mid"] > 0]
                    merged = merged.sort_values("strike")

                    if len(merged) == 0:
                        continue

                    # Calcul DTE
                    now = datetime.now()
                    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                    dte = (exp_dt - now).days

                    color = colors[idx % len(colors)]

                    fig.add_trace(go.Scatter(
                        x=merged["strike"].values,
                        y=(merged["iv_mid"] * 100).values,  # en pourcentage
                        mode="lines",
                        name=f"{exp} ({dte} DTE) σ={merged['iv_mid'].mean():.3f}",
                        line=dict(color=color, width=1, dash="dot"),
                        hovertemplate=f"Strike: %{{x:,.0f}}<br>IV: %{{y:.2f}}%<br>{exp} ({dte} DTE)<extra></extra>"
                    ))

                    progress.progress((idx + 1) / total, text=f"Fetching skew... {idx+1}/{total}")

                progress.empty()

                # Ligne verticale spot (cyan, dashed)
                fig.add_vline(
                    x=spot,
                    line=dict(color="#00FFFF", width=1, dash="dot"),
                    annotation_text=f"SPOT ${spot:,.0f}",
                    annotation_position="top",
                    annotation_font=dict(color="#00FFFF", size=10, family="monospace")
                )

                fig.update_layout(
                    title=dict(
                        text=(
                            f"<b>{symbol} VOLATILITY AT THE MID SMILE</b><br>"
                            f"<span style='font-size:11px;color:#666'>{datetime.now().strftime('%Y-%m-%d %H:%M')} — Spot: ${spot:,.2f}</span>"
                        ),
                        x=0.5, xanchor="center",
                        font=dict(size=14, color="white", family="monospace")
                    ),
                    xaxis=dict(
                        title=dict(text="STRIKE", font=dict(color="#555", size=11, family="monospace")),
                        tickfont=dict(color="#888", size=9, family="monospace"),
                        showgrid=False,
                        showline=True, linecolor="#222",
                        tickprefix="$", tickformat=",."
                    ),
                    yaxis=dict(
                        title=dict(text="VOLATILITY AT THE MID", font=dict(color="#555", size=11, family="monospace")),
                        tickfont=dict(color="#888", size=9, family="monospace"),
                        showgrid=False,
                        showline=True, linecolor="#222",
                        ticksuffix=""
                    ),
                    legend=dict(
                        x=0.02, y=0.98, xanchor="left", yanchor="top",
                        bgcolor="rgba(10,10,20,0.85)", bordercolor="#333", borderwidth=1,
                        font=dict(color="white", size=9, family="monospace")
                    ),
                    paper_bgcolor="#000", plot_bgcolor="#0a0a14",
                    width=794, height=600,
                    margin=dict(l=70, r=40, t=80, b=60)
                )

                st.plotly_chart(fig, use_container_width=False)

            # ============================================================
            # MODE GAMMA
            # ============================================================
            elif mode == "Gamma":
                # ── Résoudre l'expiration choisie ──
                # ticker.options est trié par date croissante.
                # On capped l'index au nombre d'expirations dispo.
                available_exps = ticker.options
                gamma_idx = min(gamma_exp_index, len(available_exps) - 1)
                first_exp = available_exps[gamma_idx]

                exp_dt  = datetime.strptime(first_exp, "%Y-%m-%d")
                dte     = max((exp_dt - datetime.now()).days, 0)   # ne peut pas être négatif
                exp_fmt = exp_dt.strftime("%b %d '%y")

                col2.metric("📅 Expiration", f"{exp_fmt} ({dte} DTE)")
                col3.metric("📐 Range", f"±{strike_range} pts")

                chain = ticker.option_chain(first_exp)

                progress = st.progress(0, text="Computing gamma profile...")
                profile  = compute_gamma_profile(chain.calls, chain.puts, spot, first_exp, risk_free_rate)
                progress.progress(1.0, text="Done.")
                progress.empty()

                # ── Filtrer le net GEX dans le range du spot ──
                net_full = profile["net"]
                strike_min_g = spot - strike_range
                strike_max_g = spot + strike_range
                net_filtered = net_full[(net_full.index >= strike_min_g) & (net_full.index <= strike_max_g)]
                profile["net"] = net_filtered

                # ── Déterminer gamma positif ou négatif au spot ──
                above_spot  = net_full[net_full.index >= spot].sort_index()
                cum_above   = above_spot.cumsum()
                gamma_now   = "POSITIF" if float(cum_above.iloc[0]) >= 0 else "NÉGATIF"

                # ── Helpers ──
                def fmt_val(v):
                    av = abs(v)
                    if av >= 1e9: return f"${v/1e9:+.2f} B"
                    if av >= 1e6: return f"${v/1e6:+.2f} M"
                    if av >= 1e3: return f"${v/1e3:+.1f} K"
                    return f"${v:+,.0f}"

                gamma_color = "#00cc66" if gamma_now == "POSITIF" else "#cc3333"

                # ── Layout côte à côte : graphique | tableau ──
                col_chart, col_table = st.columns([3, 1], gap="small")

                # --- Graphique (colonne gauche, pleine largeur) ---
                with col_chart:
                    fig = create_gamma_chart(profile, spot, symbol, first_exp)
                    st.plotly_chart(fig, use_container_width=True)

                # --- Tableau (colonne droite) ---
                with col_table:
                    st.markdown(f"""
                    <div style="
                        background:#0a0a0a;
                        border:1px solid #222;
                        border-radius:8px;
                        padding:16px 18px;
                        font-family:'Segoe UI', Arial, sans-serif;
                        margin-top:4px;
                    ">
                      <!-- Header : expiration + badge gamma -->
                      <div style="margin-bottom:14px;">
                        <div style="color:#aaa; font-size:12px; margin-bottom:8px;">
                          {symbol} &mdash; Exp : <b style='color:#fff'>{exp_fmt}</b> ({dte} DTE)
                        </div>
                        <span style="
                            display:inline-block;
                            background:{gamma_color}22; color:{gamma_color};
                            border:1px solid {gamma_color}; border-radius:5px;
                            padding:4px 12px; font-size:13px; font-weight:600;
                        ">GAMMA {gamma_now}</span>
                      </div>

                      <!-- Séparateur -->
                      <hr style="border:none; border-top:1px solid #222; margin:12px 0;">

                      <!-- Ligne : Vol Trigger -->
                      <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #111;">
                        <span style="color:#FFD700; font-size:15px; font-weight:600;">⚡ Vol Trigger</span>
                        <span style="color:#FFD700; font-size:15px; font-weight:600;">${profile['vol_trigger']:,.0f}</span>
                      </div>

                      <!-- Ligne : Major Wall -->
                      <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #111;">
                        <span style="color:#FF6600; font-size:15px; font-weight:600;">🏗 Major Wall</span>
                        <span style="color:#FF6600; font-size:15px; font-weight:600;">${profile['major_wall']:,.0f}</span>
                      </div>
                      <div style="display:flex; justify-content:flex-end; padding:2px 0 6px 0;">
                        <span style="color:#666; font-size:11px;">{fmt_val(net_full[profile['major_wall']])}</span>
                      </div>

                      <!-- Ligne : Call Wall -->
                      <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #111;">
                        <span style="color:#4488FF; font-size:15px; font-weight:600;">📈 Call Wall</span>
                        <span style="color:#4488FF; font-size:15px; font-weight:600;">${profile['call_wall']:,.0f}</span>
                      </div>
                      <div style="display:flex; justify-content:flex-end; padding:2px 0 6px 0;">
                        <span style="color:#666; font-size:11px;">{fmt_val(profile['call_gex'][profile['call_wall']])}</span>
                      </div>

                      <!-- Ligne : Put Wall -->
                      <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #111;">
                        <span style="color:#FF4488; font-size:15px; font-weight:600;">📉 Put Wall</span>
                        <span style="color:#FF4488; font-size:15px; font-weight:600;">${profile['put_wall']:,.0f}</span>
                      </div>
                      <div style="display:flex; justify-content:flex-end; padding:2px 0 6px 0;">
                        <span style="color:#666; font-size:11px;">{fmt_val(profile['put_gex'][profile['put_wall']])}</span>
                      </div>

                      <!-- Séparateur -->
                      <hr style="border:none; border-top:1px solid #222; margin:10px 0;">

                      <!-- Ligne : Net GEX total -->
                      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0;">
                        <span style="color:#00FFFF; font-size:13px; font-weight:600;">💰 Net GEX</span>
                        <span style="color:#00FFFF; font-size:14px; font-weight:700;">{fmt_val(profile['net_gex_oi'])}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
else:
    st.info("👈 Configure tes paramètres dans le sidebar puis clique sur **Calculer**")
