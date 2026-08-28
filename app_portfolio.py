"""
Simulador de Portafolio de Inversión — Streamlit App
=====================================================
Interfaz para armar un portafolio con activos reales
y analizar retorno esperado, volatilidad y evolución histórica.
Incluye Scanner de Mercados, Rankings y Comparación de Carteras.

Autor: Curso Python
Versión: 4.0
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from portfolio_model import PortfolioAnalyzer, ACTIVOS
from market_scanner_model import MarketScanner, PERIODOS
from dividend_scanner_model import DividendScanner
from fundamentals_scanner_model import FundamentalsScanner, PERIODOS_FUND
from multifactor_model import MultifactorScanner, PESOS, TOP_N as MF_TOP_N
from contrarian_model import ContrarianScanner, PERIODOS_CONTRARIAN, TOP_N as CT_TOP_N
from market_analysis_model import MarketAnalyzer, MERCADO_MAP, UNIVERSO, MERCADOS_DISPONIBLES

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Portafolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f7f5f0;
        color: #1a1a1a;
    }
    h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        color: #1a1a1a;
        letter-spacing: -2px;
        font-size: 2.8rem;
    }
    h2, h3 { font-family: 'Syne', sans-serif; color: #1a1a1a; }

    .stButton>button {
        background-color: #1a1a1a;
        color: #f7f5f0;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 0px;
        padding: 0.75rem 2rem;
        width: 100%;
        font-size: 0.95rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .stButton>button:hover { background-color: #333; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e0ddd5;
        padding: 1.5rem;
        margin: 0.3rem 0;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
    }
    .positive { color: #1a7a4a; }
    .negative { color: #c0392b; }
    .neutral  { color: #1a1a1a; }
    .warning-text {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        margin-top: 0.5rem;
    }

    /* ── Scanner styles ── */
    .scanner-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
        border-bottom: 3px solid #1a1a1a;
        margin-bottom: 1rem;
    }
    .scanner-period-label {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.3rem;
    }
    .rank-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.75rem;
        border-bottom: 1px solid #eee;
        font-size: 0.88rem;
    }
    .rank-row:hover { background: #f0ede6; }
    .rank-num {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.75rem;
        color: #ccc;
        width: 1.5rem;
    }
    .rank-ticker {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        flex: 1;
        padding-left: 0.5rem;
    }
    .rank-change-loser  { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #c0392b; }
    .rank-change-gainer { font-family: 'DM Sans', sans-serif; font-weight: 500; color: #1a7a4a; }
    .scanner-card {
        background: #ffffff;
        border: 1px solid #e0ddd5;
        padding: 1rem 1rem 0.5rem 1rem;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────

def _render_ranking_card(titulo: str, df_dict: dict, modo: str = "losers"):
    """Renderiza el bloque completo de un mercado con 4 períodos en columnas."""
    st.markdown(f'<div class="scanner-header">{titulo}</div>', unsafe_allow_html=True)

    periodos_nombres = list(PERIODOS.keys())
    cols = st.columns(4)
    change_class = "rank-change-loser" if modo == "losers" else "rank-change-gainer"

    for col, periodo in zip(cols, periodos_nombres):
        df = df_dict.get(periodo)
        with col:
            st.markdown(f'<div class="scanner-period-label">⏱ {periodo}</div>', unsafe_allow_html=True)
            if df is None or df.empty:
                st.caption("Sin datos disponibles.")
            else:
                rows_html = ""
                for i, row in df.iterrows():
                    ticker = str(row["Ticker"]).replace(".BA", "")
                    variacion = row["Variación (%)"]
                    rows_html += f"""
                    <div class="rank-row">
                        <span class="rank-num">{i+1}</span>
                        <span class="rank-ticker">{ticker}</span>
                        <span class="{change_class}">{variacion:+.1f}%</span>
                    </div>"""

                st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)


def _render_bar_chart(titulo: str, df_dict: dict, periodo_key: str, modo: str = "losers"):
    """Bar chart horizontal del ranking para un período dado."""
    df = df_dict.get(periodo_key)
    if df is None or df.empty:
        return

    df_plot = df.copy()
    df_plot["Ticker"] = df_plot["Ticker"].str.replace(".BA", "", regex=False)

    if modo == "losers":
        colors = [f"rgba(192,57,43,{0.4 + 0.06 * i})" for i in range(len(df_plot))]
    else:
        colors = [f"rgba(26,122,74,{0.4 + 0.06 * i})" for i in range(len(df_plot))]

    fig = go.Figure(go.Bar(
        x=df_plot["Variación (%)"],
        y=df_plot["Ticker"],
        orientation="h",
        marker_color=colors,
        text=df_plot["Variación (%)"].apply(lambda v: f"{v:+.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{titulo} — {periodo_key}",
        xaxis_title="Variación (%)",
        yaxis=dict(autorange="reversed"),
        height=400,
        template="plotly_white",
        margin=dict(l=10, r=60, t=40, b=30),
        xaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, width='stretch')


# ─────────────────────────────────────────────
# CARGA CACHEADA DEL ANALIZADOR DE MERCADO
# ─────────────────────────────────────────────
# st.cache_resource porque MarketAnalyzer es un objeto con estado (guarda el
# DataFrame de precios de todo el universo). TTL de 24hs: los datos se
# recalculan "a la fecha de hoy" una vez por día, no en cada click del
# usuario — evita descargar ~370 tickers en cada interacción.
@st.cache_resource(ttl=60 * 60 * 24, show_spinner=False)
def cargar_market_analyzer(periodo: str = "3y") -> MarketAnalyzer:
    return MarketAnalyzer(periodo)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Configurar Portafolio")
    st.markdown("---")

    nombres_activos = list(ACTIVOS.keys())
    seleccionados = st.multiselect(
        "Seleccioná los activos",
        options=nombres_activos,
        default=["Apple (AAPL)", "MercadoLibre (MELI)", "Tesla (TSLA)"],
    )

    periodo = st.selectbox(
        "Período histórico",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
        format_func=lambda x: {
            "6mo": "6 meses", "1y": "1 año", "2y": "2 años", "5y": "5 años"
        }[x],
    )

    st.markdown("---")

    pesos_input = {}
    if seleccionados:
        st.markdown("**Pesos del portafolio (%)**")
        peso_default = int(100 / len(seleccionados))
        total = 0
        for nombre in seleccionados:
            ticker = ACTIVOS[nombre]
            peso = st.slider(
                f"{nombre.split('(')[0].strip()}",
                min_value=0, max_value=100,
                value=peso_default, step=5,
                key=f"peso_{ticker}"
            )
            pesos_input[ticker] = peso
            total += peso

        if total != 100:
            st.warning(f"Los pesos suman {total}%. Se normalizarán automáticamente a 100%.")
        else:
            st.success("✓ Los pesos suman 100%")

    st.markdown("---")
    analizar = st.button("▶  ANALIZAR PORTAFOLIO")

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.markdown("# Simulador de Portafolio")
st.markdown("Datos reales desde Yahoo Finance · Curso Python")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
(tab_portfolio, tab_comparar, tab_scanner, tab_dividendos, tab_per, tab_roe, tab_margen,
 tab_mf, tab_contrarian, tab_mercado, tab_simulador) = st.tabs([
    "📈 Portafolio",
    "⚖️ Comparar Carteras",
    "📉 Scanner de Mercados",
    "💰 Ranking de Dividendos",
    "📊 Ranking PER",
    "💹 Ranking ROE",
    "📉 Ranking Márgenes",
    "🏆 Smart Ranking",
    "🎯 Contrarian Ranking",
    "🧭 Análisis de Mercado",
    "🧪 Simulador Personalizado",
])

# ══════════════════════════════════════════════
# TAB 1 — PORTAFOLIO
# ══════════════════════════════════════════════
with tab_portfolio:

    if analizar:
        if not seleccionados:
            st.warning("Seleccioná al menos un activo para continuar.")
        elif sum(pesos_input.values()) == 0:
            st.warning("Asigná pesos mayores a 0 para continuar.")
        else:
            tickers   = [ACTIVOS[n] for n in seleccionados]
            pesos_dec = {t: p / 100 for t, p in pesos_input.items()}

            with st.spinner("Descargando datos desde Yahoo Finance..."):
                try:
                    analyzer  = PortfolioAnalyzer(tickers, periodo)
                    resultado = analyzer.analizar_portafolio(pesos_dec)
                    metricas  = analyzer.metricas_individuales()
                    st.session_state.analyzer      = analyzer
                    st.session_state.resultado     = resultado
                    st.session_state.metricas      = metricas
                    st.session_state.seleccionados = seleccionados
                    st.session_state.pesos_input   = pesos_input
                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")
                    st.stop()

    if "resultado" in st.session_state:
        r        = st.session_state.resultado
        ana      = st.session_state.analyzer
        metricas = st.session_state.metricas

        col1, col2, col3, col4 = st.columns(4)
        ret_color  = "positive" if r["retorno_anual"] >= 0 else "negative"
        draw_color = "negative" if r["max_drawdown"]  < -15 else "neutral"

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Retorno Anual Est.</p>
                <p class="metric-value {ret_color}">{r['retorno_anual']:+.1f}%</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Volatilidad Anual</p>
                <p class="metric-value neutral">{r['volatilidad']:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Sharpe Ratio</p>
                <p class="metric-value neutral">{r['sharpe']:.2f}</p>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Máx. Drawdown</p>
                <p class="metric-value {draw_color}">{r['max_drawdown']:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Evolución", "📊 Activos Individuales", "🔗 Correlaciones"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=r["fechas"], y=r["evolucion"],
                mode="lines", name="Mi Portafolio",
                line=dict(color="#1a1a1a", width=3),
                fill="tozeroy", fillcolor="rgba(26,26,26,0.05)",
            ))
            fig.add_hline(y=100, line_dash="dot", line_color="#aaa",
                          annotation_text="Base 100", annotation_position="right")
            fig.update_layout(
                title="Evolución del Portafolio (Base 100)",
                xaxis_title="Fecha", yaxis_title="Valor (Base 100)",
                height=450, template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown('<p class="warning-text">* Retorno histórico no garantiza rendimiento futuro. Solo con fines educativos.</p>',
                        unsafe_allow_html=True)

        with tab2:
            col_a, col_b = st.columns([2, 1])

            with col_a:
                df_scatter = metricas.reset_index()
                df_scatter = df_scatter.rename(columns={df_scatter.columns[0]: "Ticker"})
                fig2 = px.scatter(
                    df_scatter,
                    x="Volatilidad Anual (%)", y="Retorno Anual (%)",
                    text="Ticker",
                    size=[20] * len(df_scatter),
                    color="Sharpe Ratio", color_continuous_scale="RdYlGn",
                    title="Retorno vs Riesgo por Activo",
                )
                fig2.update_traces(textposition="top center", marker=dict(sizemode="diameter"))
                fig2.add_hline(y=0, line_dash="dot", line_color="#ccc")
                fig2.update_layout(height=400, template="plotly_white")
                st.plotly_chart(fig2, width='stretch')

            with col_b:
                st.markdown("### Tabla de Métricas")
                st.dataframe(
                    metricas.style.format({
                        "Retorno Anual (%)":     "{:+.2f}%",
                        "Volatilidad Anual (%)": "{:.2f}%",
                        "Sharpe Ratio":          "{:.2f}",
                    }),
                    width='stretch',
                )

            pesos_norm  = st.session_state.pesos_input
            total_pesos = sum(pesos_norm.values())
            labels = [n.split("(")[0].strip() for n in st.session_state.seleccionados]
            values = [pesos_norm[ACTIVOS[n]] / total_pesos * 100 for n in st.session_state.seleccionados]

            fig3 = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.45,
                marker=dict(colors=px.colors.qualitative.G10),
            ))
            fig3.update_layout(title="Composición del Portafolio", height=350, template="plotly_white")
            st.plotly_chart(fig3, width='stretch')

        with tab3:
            if len(ana.retornos.columns) > 1:
                corr = ana.matriz_correlacion()
                fig4 = px.imshow(
                    corr, text_auto=True,
                    color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                    title="Matriz de Correlación entre Activos", aspect="auto",
                )
                fig4.update_layout(height=450, template="plotly_white")
                st.plotly_chart(fig4, width='stretch')
                st.info("💡 Correlaciones cercanas a -1 indican activos que se mueven en sentidos opuestos, lo que **reduce el riesgo** del portafolio.")
            else:
                st.info("Seleccioná más de un activo para ver correlaciones.")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color:#bbb;">
            <p style="font-size:5rem; margin:0;">📈</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.2rem; color:#aaa; margin-top:1rem;">
                Elegí los activos y los pesos en el panel izquierdo<br>
                y hacé clic en <strong style="color:#1a1a1a;">ANALIZAR PORTAFOLIO</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — COMPARACIÓN DE CARTERAS
# ══════════════════════════════════════════════
with tab_comparar:
    st.markdown("### ⚖️ Comparación de Carteras")
    st.markdown(
        "Armá **dos carteras** distintas y compará su evolución histórica, "
        "retorno, volatilidad y Sharpe Ratio lado a lado."
    )
    st.markdown("---")

    nombres_activos_cmp = list(ACTIVOS.keys())
    PALETA_A = "#2563eb"   # azul — Cartera A
    PALETA_B = "#e11d48"   # rojo — Cartera B

    # ── Configuración de las dos carteras ──────────────────────
    col_cfg_a, col_sep, col_cfg_b = st.columns([5, 1, 5])

    with col_cfg_a:
        st.markdown(f'<div style="background:#eff6ff; border-left:4px solid {PALETA_A}; padding:0.75rem 1rem; font-family:Syne,sans-serif; font-weight:800; font-size:1.1rem; color:{PALETA_A}; margin-bottom:0.75rem;">🔵 Cartera A</div>', unsafe_allow_html=True)
        sel_a = st.multiselect(
            "Activos Cartera A",
            options=nombres_activos_cmp,
            default=["Apple (AAPL)", "Amazon (AMZN)", "Google (GOOGL)"],
            key="cmp_sel_a",
        )
        pesos_a_input = {}
        if sel_a:
            peso_def_a = int(100 / len(sel_a))
            total_a = 0
            st.markdown("**Pesos Cartera A (%)**")
            for nombre in sel_a:
                ticker = ACTIVOS[nombre]
                p = st.slider(
                    nombre.split("(")[0].strip(),
                    min_value=0, max_value=100,
                    value=peso_def_a, step=5,
                    key=f"cmp_a_{ticker}"
                )
                pesos_a_input[ticker] = p
                total_a += p
            color_msg_a = "success" if total_a == 100 else "warning"
            getattr(st, color_msg_a)(f"Suma: {total_a}%" + (" ✓" if total_a == 100 else " → se normalizará"))

    with col_sep:
        st.markdown("<div style='text-align:center; padding-top:4rem; font-size:2rem; color:#ccc;'>VS</div>", unsafe_allow_html=True)

    with col_cfg_b:
        st.markdown(f'<div style="background:#fff1f2; border-left:4px solid {PALETA_B}; padding:0.75rem 1rem; font-family:Syne,sans-serif; font-weight:800; font-size:1.1rem; color:{PALETA_B}; margin-bottom:0.75rem;">🔴 Cartera B</div>', unsafe_allow_html=True)
        sel_b = st.multiselect(
            "Activos Cartera B",
            options=nombres_activos_cmp,
            default=["JPMorgan (JPM)", "Coca-Cola (KO)", "Exxon (XOM)"],
            key="cmp_sel_b",
        )
        pesos_b_input = {}
        if sel_b:
            peso_def_b = int(100 / len(sel_b))
            total_b = 0
            st.markdown("**Pesos Cartera B (%)**")
            for nombre in sel_b:
                ticker = ACTIVOS[nombre]
                p = st.slider(
                    nombre.split("(")[0].strip(),
                    min_value=0, max_value=100,
                    value=peso_def_b, step=5,
                    key=f"cmp_b_{ticker}"
                )
                pesos_b_input[ticker] = p
                total_b += p
            color_msg_b = "success" if total_b == 100 else "warning"
            getattr(st, color_msg_b)(f"Suma: {total_b}%" + (" ✓" if total_b == 100 else " → se normalizará"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Período ────────────────────────────────────────────────
    cmp_c1, cmp_c2 = st.columns([2, 4])
    with cmp_c1:
        periodo_cmp = st.selectbox(
            "Período histórico",
            options=["6mo", "1y", "2y", "5y"],
            index=1,
            format_func=lambda x: {
                "6mo": "6 meses", "1y": "1 año", "2y": "2 años", "5y": "5 años"
            }[x],
            key="cmp_periodo",
        )

    comparar_btn = st.button("⚖️  COMPARAR CARTERAS", key="btn_comparar")

    if comparar_btn:
        if not sel_a or not sel_b:
            st.warning("Seleccioná activos en ambas carteras para comparar.")
        elif sum(pesos_a_input.values()) == 0 or sum(pesos_b_input.values()) == 0:
            st.warning("Asigná pesos mayores a 0 en ambas carteras.")
        else:
            tickers_a = [ACTIVOS[n] for n in sel_a]
            tickers_b = [ACTIVOS[n] for n in sel_b]
            pesos_dec_a = {t: p / 100 for t, p in pesos_a_input.items()}
            pesos_dec_b = {t: p / 100 for t, p in pesos_b_input.items()}

            all_tickers = list(set(tickers_a + tickers_b))

            with st.spinner("Descargando datos de ambas carteras..."):
                try:
                    # Descargamos todos los tickers juntos para eficiencia
                    ana_a = PortfolioAnalyzer(tickers_a, periodo_cmp)
                    res_a = ana_a.analizar_portafolio(pesos_dec_a)

                    ana_b = PortfolioAnalyzer(tickers_b, periodo_cmp)
                    res_b = ana_b.analizar_portafolio(pesos_dec_b)

                    st.session_state["cmp_res_a"]    = res_a
                    st.session_state["cmp_res_b"]    = res_b
                    st.session_state["cmp_ana_a"]    = ana_a
                    st.session_state["cmp_ana_b"]    = ana_b
                    st.session_state["cmp_sel_a_lbl"] = sel_a
                    st.session_state["cmp_sel_b_lbl"] = sel_b
                    st.session_state["cmp_pesos_a"]  = pesos_a_input
                    st.session_state["cmp_pesos_b"]  = pesos_b_input

                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")

    # ── Mostrar resultados ──────────────────────────────────────
    if "cmp_res_a" in st.session_state and "cmp_res_b" in st.session_state:
        res_a = st.session_state["cmp_res_a"]
        res_b = st.session_state["cmp_res_b"]
        sel_a_lbl = st.session_state.get("cmp_sel_a_lbl", [])
        sel_b_lbl = st.session_state.get("cmp_sel_b_lbl", [])

        st.markdown("---")

        # ── KPIs comparativos ──────────────────────────────────
        st.markdown('<div class="scanner-header">📊 Comparación de Métricas</div>', unsafe_allow_html=True)

        metricas_labels = ["Retorno Anual Est.", "Volatilidad Anual", "Sharpe Ratio", "Máx. Drawdown"]
        vals_a = [res_a["retorno_anual"], res_a["volatilidad"], res_a["sharpe"], res_a["max_drawdown"]]
        vals_b = [res_b["retorno_anual"], res_b["volatilidad"], res_b["sharpe"], res_b["max_drawdown"]]
        formatos = ["{:+.2f}%", "{:.2f}%", "{:.2f}", "{:+.2f}%"]

        cols_kpi = st.columns(4)
        for col_k, lbl, va, vb, fmt in zip(cols_kpi, metricas_labels, vals_a, vals_b, formatos):
            # Para drawdown y volatilidad, "mejor" es el menor valor en magnitud
            if lbl in ["Máx. Drawdown", "Volatilidad Anual"]:
                winner = "A" if va > vb else ("B" if vb > va else "=")  # mayor drawdown = peor
                if lbl == "Volatilidad Anual":
                    winner = "A" if va < vb else ("B" if vb < va else "=")
            else:
                winner = "A" if va > vb else ("B" if vb > va else "=")

            color_a = PALETA_A if winner == "A" else ("#555" if winner == "=" else "#aaa")
            color_b = PALETA_B if winner == "B" else ("#555" if winner == "=" else "#aaa")
            badge = "🏆 " if winner != "=" else ""

            with col_k:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{lbl}</p>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
                        <div style="text-align:center; flex:1;">
                            <div style="font-size:0.65rem; letter-spacing:1px; color:{PALETA_A}; font-weight:700;">🔵 CARTERA A</div>
                            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:{color_a};">
                                {badge if winner=='A' else ''}{fmt.format(va)}
                            </div>
                        </div>
                        <div style="color:#ccc; font-size:1rem; padding:0 0.5rem;">|</div>
                        <div style="text-align:center; flex:1;">
                            <div style="font-size:0.65rem; letter-spacing:1px; color:{PALETA_B}; font-weight:700;">🔴 CARTERA B</div>
                            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:{color_b};">
                                {badge if winner=='B' else ''}{fmt.format(vb)}
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gráfico de evolución comparada ─────────────────────
        st.markdown('<div class="scanner-header">📈 Evolución Histórica Comparada (Base 100)</div>', unsafe_allow_html=True)

        ev_a = res_a["evolucion"]
        ev_b = res_b["evolucion"]

        fig_cmp = go.Figure()

        fig_cmp.add_trace(go.Scatter(
            x=ev_a.index, y=ev_a.values,
            mode="lines", name="🔵 Cartera A",
            line=dict(color=PALETA_A, width=2.5),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=ev_b.index, y=ev_b.values,
            mode="lines", name="🔴 Cartera B",
            line=dict(color=PALETA_B, width=2.5),
        ))
        fig_cmp.add_hline(y=100, line_dash="dot", line_color="#ccc",
                          annotation_text="Base 100", annotation_position="right")

        # Área de diferencia
        idx_comun = ev_a.index.intersection(ev_b.index)
        if len(idx_comun) > 0:
            ev_a_c = ev_a.reindex(idx_comun)
            ev_b_c = ev_b.reindex(idx_comun)
            diferencia = ev_a_c - ev_b_c
            fig_cmp.add_trace(go.Scatter(
                x=idx_comun, y=diferencia.values,
                mode="lines", name="Diferencia (A − B)",
                line=dict(color="#888", width=1, dash="dot"),
                yaxis="y2",
                visible="legendonly",
            ))

        fig_cmp.update_layout(
            title="Evolución del Portafolio — Cartera A vs Cartera B",
            xaxis_title="Fecha",
            yaxis_title="Valor (Base 100)",
            yaxis2=dict(
                title="Diferencia (A − B)",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=True,
                zerolinecolor="#ddd",
            ),
            height=480,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cmp, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Drawdown comparado ─────────────────────────────────
        with st.expander("📉 Ver Drawdown Comparado"):
            ev_a_dd = ev_a / ev_a.cummax() * 100 - 100
            ev_b_dd = ev_b / ev_b.cummax() * 100 - 100

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=ev_a.index, y=ev_a_dd.values,
                mode="lines", name="🔵 Cartera A — Drawdown",
                line=dict(color=PALETA_A, width=1.5),
                fill="tozeroy", fillcolor=f"rgba(37,99,235,0.08)",
            ))
            fig_dd.add_trace(go.Scatter(
                x=ev_b.index, y=ev_b_dd.values,
                mode="lines", name="🔴 Cartera B — Drawdown",
                line=dict(color=PALETA_B, width=1.5),
                fill="tozeroy", fillcolor=f"rgba(225,29,72,0.08)",
            ))
            fig_dd.update_layout(
                title="Drawdown histórico comparado (%)",
                yaxis_title="Drawdown (%)",
                height=350,
                template="plotly_white",
                hovermode="x unified",
                yaxis=dict(ticksuffix="%"),
            )
            st.plotly_chart(fig_dd, width='stretch')

        # ── Retornos anuales por año ────────────────────────────
        with st.expander("📅 Retornos Anuales por Año"):
            ev_a_s = ev_a.copy()
            ev_b_s = ev_b.copy()

            # Calcular retorno de cada año calendario
            def _retornos_por_anio(ev_serie):
                ev_serie.index = pd.to_datetime(ev_serie.index)
                anios = ev_serie.index.year.unique()
                retornos = {}
                for anio in anios:
                    sub = ev_serie[ev_serie.index.year == anio]
                    if len(sub) >= 2:
                        ret_anio = (sub.iloc[-1] / sub.iloc[0] - 1) * 100
                        retornos[anio] = round(ret_anio, 2)
                return retornos

            ret_anios_a = _retornos_por_anio(ev_a)
            ret_anios_b = _retornos_por_anio(ev_b)
            anios_todos = sorted(set(list(ret_anios_a.keys()) + list(ret_anios_b.keys())))

            if anios_todos:
                fig_bar_anios = go.Figure()
                fig_bar_anios.add_trace(go.Bar(
                    x=anios_todos,
                    y=[ret_anios_a.get(a, 0) for a in anios_todos],
                    name="🔵 Cartera A",
                    marker_color=PALETA_A,
                    text=[f"{ret_anios_a.get(a, 0):+.1f}%" for a in anios_todos],
                    textposition="outside",
                ))
                fig_bar_anios.add_trace(go.Bar(
                    x=anios_todos,
                    y=[ret_anios_b.get(a, 0) for a in anios_todos],
                    name="🔴 Cartera B",
                    marker_color=PALETA_B,
                    text=[f"{ret_anios_b.get(a, 0):+.1f}%" for a in anios_todos],
                    textposition="outside",
                ))
                fig_bar_anios.update_layout(
                    barmode="group",
                    title="Retorno por Año Calendario (%)",
                    yaxis_title="Retorno (%)",
                    height=380,
                    template="plotly_white",
                    yaxis=dict(ticksuffix="%"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_bar_anios, width='stretch')

        # ── Composición de las carteras ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="scanner-header">🥧 Composición de las Carteras</div>', unsafe_allow_html=True)

        col_pie_a, col_pie_b = st.columns(2)
        pesos_a_stored = st.session_state.get("cmp_pesos_a", {})
        pesos_b_stored = st.session_state.get("cmp_pesos_b", {})

        with col_pie_a:
            total_pa = sum(pesos_a_stored.values()) or 1
            labels_a = [n.split("(")[0].strip() for n in sel_a_lbl]
            values_a = [pesos_a_stored.get(ACTIVOS[n], 0) / total_pa * 100 for n in sel_a_lbl]
            fig_pie_a = go.Figure(go.Pie(
                labels=labels_a, values=values_a, hole=0.4,
                marker=dict(colors=px.colors.qualitative.Pastel1),
            ))
            fig_pie_a.update_layout(
                title=f"<span style='color:{PALETA_A};'>🔵 Cartera A</span>",
                height=300, template="plotly_white",
            )
            st.plotly_chart(fig_pie_a, width='stretch')

        with col_pie_b:
            total_pb = sum(pesos_b_stored.values()) or 1
            labels_b = [n.split("(")[0].strip() for n in sel_b_lbl]
            values_b = [pesos_b_stored.get(ACTIVOS[n], 0) / total_pb * 100 for n in sel_b_lbl]
            fig_pie_b = go.Figure(go.Pie(
                labels=labels_b, values=values_b, hole=0.4,
                marker=dict(colors=px.colors.qualitative.Pastel2),
            ))
            fig_pie_b.update_layout(
                title=f"<span style='color:{PALETA_B};'>🔴 Cartera B</span>",
                height=300, template="plotly_white",
            )
            st.plotly_chart(fig_pie_b, width='stretch')

        # ── Tabla de retornos por activo ───────────────────────
        with st.expander("📋 Ver métricas individuales por activo"):
            tab_ind_a, tab_ind_b = st.tabs(["🔵 Cartera A", "🔴 Cartera B"])
            with tab_ind_a:
                ana_a_obj = st.session_state.get("cmp_ana_a")
                if ana_a_obj:
                    met_a = ana_a_obj.metricas_individuales()
                    st.dataframe(
                        met_a.style.format({
                            "Retorno Anual (%)":     "{:+.2f}%",
                            "Volatilidad Anual (%)": "{:.2f}%",
                            "Sharpe Ratio":          "{:.2f}",
                        }),
                        width='stretch',
                    )
            with tab_ind_b:
                ana_b_obj = st.session_state.get("cmp_ana_b")
                if ana_b_obj:
                    met_b = ana_b_obj.metricas_individuales()
                    st.dataframe(
                        met_b.style.format({
                            "Retorno Anual (%)":     "{:+.2f}%",
                            "Volatilidad Anual (%)": "{:.2f}%",
                            "Sharpe Ratio":          "{:.2f}",
                        }),
                        width='stretch',
                    )

        st.markdown('<p class="warning-text">* Retorno histórico no garantiza rendimiento futuro. Solo con fines educativos.</p>',
                    unsafe_allow_html=True)

    elif not comparar_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">⚖️</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Configurá las dos carteras con sus activos y pesos,<br>
                luego hacé clic en <strong style="color:#1a1a1a;">COMPARAR CARTERAS</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — SCANNER DE MERCADOS
# ══════════════════════════════════════════════
with tab_scanner:
    st.markdown("### 🔍 Scanner de Mercados")
    st.markdown("Analizá las acciones con mejor y peor rendimiento en los últimos 1, 3, 6 y 12 meses.")
    st.markdown("---")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

    with ctrl1:
        mercado_sel = st.radio(
            "Mercado:",
            options=["Mercado Argentino (ByMA)", "Mercado de Nueva York (NYSE/NASDAQ)", "Ambos mercados"],
            horizontal=True,
            key="scanner_mercado",
        )

    with ctrl2:
        modo_sel = st.radio(
            "Tipo de ranking:",
            options=["losers", "gainers"],
            format_func=lambda x: "📉 Mayores caídas" if x == "losers" else "📈 Mayores subidas",
            horizontal=False,
            key="scanner_modo",
        )

    with ctrl3:
        vista_sel = st.selectbox(
            "Vista",
            options=["Ranking tablas", "Gráfico de barras"],
            key="scanner_vista",
        )

    if vista_sel == "Gráfico de barras":
        periodo_grafico = st.selectbox(
            "Período para el gráfico",
            options=list(PERIODOS.keys()),
            key="scanner_periodo_grafico",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    escanear_btn = st.button("🔍  ESCANEAR MERCADO", key="btn_escanear")

    if escanear_btn:
        mercados_a_correr = []
        if mercado_sel == "Mercado Argentino (ByMA)":
            mercados_a_correr = [("byma", "🇦🇷 Mercado Argentino (ByMA)")]
        elif mercado_sel == "Mercado de Nueva York (NYSE/NASDAQ)":
            mercados_a_correr = [("nyse", "🇺🇸 NYSE / NASDAQ")]
        else:
            mercados_a_correr = [
                ("byma", "🇦🇷 Mercado Argentino (ByMA)"),
                ("nyse", "🇺🇸 NYSE / NASDAQ"),
            ]

        resultados_scanner = {}

        for mercado_key, mercado_label in mercados_a_correr:
            with st.spinner(f"Descargando datos de {mercado_label}... (puede tardar ~30 seg)"):
                try:
                    scanner = MarketScanner(mercado=mercado_key)
                    prog_bar = st.progress(0, text=f"Analizando {mercado_label}...")

                    def _prog(val, label=mercado_label, bar=prog_bar):
                        bar.progress(val, text=f"Procesando {label}... {int(val*100)}%")

                    resultados = scanner.escanear_todos_periodos(
                        modo=modo_sel,
                        progress_callback=_prog,
                    )
                    prog_bar.empty()
                    resultados_scanner[mercado_key] = (mercado_label, resultados)

                except Exception as e:
                    st.error(f"Error al escanear {mercado_label}: {e}")

        if resultados_scanner:
            st.session_state.scanner_resultados  = resultados_scanner
            st.session_state.scanner_vista_val   = vista_sel
            st.session_state.scanner_modo_val    = modo_sel
            if vista_sel == "Gráfico de barras":
                st.session_state.scanner_periodo_val = periodo_grafico

    if "scanner_resultados" in st.session_state:
        resultados_scanner = st.session_state.scanner_resultados
        vista              = st.session_state.get("scanner_vista_val", "Ranking tablas")
        modo_guardado      = st.session_state.get("scanner_modo_val", "losers")

        if modo_guardado == "gainers":
            st.success("📈 Mostrando: **Top 10 Mayores Subidas**")
        else:
            st.error("📉 Mostrando: **Top 10 Mayores Caídas**")

        for mercado_key, (mercado_label, df_dict) in resultados_scanner.items():

            if vista == "Ranking tablas":
                _render_ranking_card(mercado_label, df_dict, modo=modo_guardado)
            else:
                periodo_g = st.session_state.get("scanner_periodo_val", "1 mes")
                _render_bar_chart(mercado_label, df_dict, periodo_g, modo=modo_guardado)

            with st.expander(f"📋 Ver datos completos — {mercado_label}"):
                tab_cols = st.tabs(list(PERIODOS.keys()))
                for tab_col, periodo_nombre in zip(tab_cols, PERIODOS.keys()):
                    with tab_col:
                        df = df_dict.get(periodo_nombre)
                        if df is not None and not df.empty:
                            df_display = df.copy()
                            df_display["Ticker"] = df_display["Ticker"].str.replace(".BA", "", regex=False)
                            st.dataframe(
                                df_display.style.format({"Variación (%)": "{:+.2f}%"}),
                                width='stretch',
                                hide_index=True,
                            )
                        else:
                            st.caption("Sin datos para este período.")

            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="warning-text">* Los datos provienen de Yahoo Finance. Algunos símbolos de ByMA pueden no estar disponibles. Solo con fines educativos.</p>',
                    unsafe_allow_html=True)

    elif not escanear_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">🔍</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado, el tipo de ranking y hacé clic en<br>
                <strong style="color:#1a1a1a;">ESCANEAR MERCADO</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — RANKING DE DIVIDENDOS
# ══════════════════════════════════════════════
with tab_dividendos:
    st.markdown("### 💰 Ranking de Dividendos")
    st.markdown("Compará qué acciones pagan más y menos dividendos según el **Dividend Yield** anual.")
    st.markdown("---")

    div_ctrl1, div_ctrl2, div_ctrl3 = st.columns([2, 1, 1])

    with div_ctrl1:
        div_mercado = st.radio(
            "Mercado:",
            options=["NYSE / NASDAQ", "Mercado Argentino (ByMA)"],
            horizontal=True,
            key="div_mercado",
        )

    with div_ctrl2:
        div_periodo_dias = st.selectbox(
            "Período de dividendos:",
            options=[365, 180, 90],
            format_func=lambda x: {365: "Últimos 12 meses", 180: "Últimos 6 meses", 90: "Últimos 3 meses"}[x],
            key="div_periodo",
        )

    with div_ctrl3:
        div_vista = st.selectbox(
            "Vista:",
            options=["Tablas", "Gráfico de barras"],
            key="div_vista",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    div_btn = st.button("💰  ESCANEAR DIVIDENDOS", key="btn_dividendos")

    if div_btn:
        mercado_key = "byma" if "ByMA" in div_mercado else "nyse"
        with st.spinner("Descargando datos de dividendos... (puede tardar ~1 min)"):
            try:
                scanner_div = DividendScanner(mercado=mercado_key)
                prog_div = st.progress(0, text="Analizando dividendos...")

                def _prog_div(val):
                    prog_div.progress(val, text=f"Analizando dividendos... {int(val*100)}%")

                resultados_div = scanner_div.escanear(dias=div_periodo_dias, progress_callback=_prog_div)
                prog_div.empty()

                st.session_state.div_resultados  = resultados_div
                st.session_state.div_mercado_lbl = div_mercado
                st.session_state.div_vista_val   = div_vista

            except Exception as e:
                st.error(f"Error al escanear dividendos: {e}")

    if "div_resultados" in st.session_state:
        res      = st.session_state.div_resultados
        lbl      = st.session_state.get("div_mercado_lbl", "")
        vista_d  = st.session_state.get("div_vista_val", "Tablas")

        df_top    = res.get("top")
        df_bottom = res.get("bottom")
        df_todos  = res.get("todos")

        if df_top is None or df_top.empty:
            st.warning("No se encontraron datos de dividendos para este mercado y período.")
        else:
            if vista_d == "Tablas":
                col_top, col_bot = st.columns(2)

                with col_top:
                    st.markdown(
                        '<div class="scanner-header" style="color:#1a7a4a;">📈 Mayores Pagadores de Dividendos</div>',
                        unsafe_allow_html=True,
                    )
                    rows_html = ""
                    for i, row in df_top.iterrows():
                        ticker = str(row["Ticker"]).replace(".BA", "")
                        nombre = str(row.get("Nombre", ticker))[:28]
                        yld    = row["Dividend_Yield (%)"]
                        rows_html += f"""
                        <div class="rank-row">
                            <span class="rank-num">{i+1}</span>
                            <span class="rank-ticker">{ticker}</span>
                            <span style="font-size:0.75rem; color:#888; flex:2; padding: 0 0.5rem;">{nombre}</span>
                            <span class="rank-change-gainer">{yld:.2f}%</span>
                        </div>"""
                    st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

                with col_bot:
                    st.markdown(
                        '<div class="scanner-header" style="color:#c0392b;">📉 Menores Pagadores de Dividendos</div>',
                        unsafe_allow_html=True,
                    )
                    if df_bottom is None or df_bottom.empty:
                        st.caption("Sin datos suficientes.")
                    else:
                        rows_html = ""
                        for i, row in df_bottom.iterrows():
                            ticker = str(row["Ticker"]).replace(".BA", "")
                            nombre = str(row.get("Nombre", ticker))[:28]
                            yld    = row["Dividend_Yield (%)"]
                            rows_html += f"""
                            <div class="rank-row">
                                <span class="rank-num">{i+1}</span>
                                <span class="rank-ticker">{ticker}</span>
                                <span style="font-size:0.75rem; color:#888; flex:2; padding: 0 0.5rem;">{nombre}</span>
                                <span class="rank-change-loser">{yld:.2f}%</span>
                            </div>"""
                        st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

            else:
                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    if df_top is not None and not df_top.empty:
                        df_plot = df_top.copy()
                        df_plot["Ticker"] = df_plot["Ticker"].str.replace(".BA", "", regex=False)
                        colors = [f"rgba(26,122,74,{0.4 + 0.055*i})" for i in range(len(df_plot))]
                        fig_top = go.Figure(go.Bar(
                            x=df_plot["Dividend_Yield (%)"],
                            y=df_plot["Ticker"],
                            orientation="h",
                            marker_color=colors,
                            text=df_plot["Dividend_Yield (%)"].apply(lambda v: f"{v:.2f}%"),
                            textposition="outside",
                        ))
                        fig_top.update_layout(
                            title="📈 Mayores Dividend Yield",
                            xaxis_title="Dividend Yield (%)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            template="plotly_white",
                            margin=dict(l=10, r=70, t=40, b=30),
                            xaxis=dict(ticksuffix="%"),
                        )
                        st.plotly_chart(fig_top, width="stretch")

                with col_g2:
                    if df_bottom is not None and not df_bottom.empty:
                        df_plot2 = df_bottom.copy()
                        df_plot2["Ticker"] = df_plot2["Ticker"].str.replace(".BA", "", regex=False)
                        colors2 = [f"rgba(192,57,43,{0.35 + 0.065*i})" for i in range(len(df_plot2))]
                        fig_bot = go.Figure(go.Bar(
                            x=df_plot2["Dividend_Yield (%)"],
                            y=df_plot2["Ticker"],
                            orientation="h",
                            marker_color=colors2,
                            text=df_plot2["Dividend_Yield (%)"].apply(lambda v: f"{v:.2f}%"),
                            textposition="outside",
                        ))
                        fig_bot.update_layout(
                            title="📉 Menores Dividend Yield (pero positivos)",
                            xaxis_title="Dividend Yield (%)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            template="plotly_white",
                            margin=dict(l=10, r=70, t=40, b=30),
                            xaxis=dict(ticksuffix="%"),
                        )
                        st.plotly_chart(fig_bot, width="stretch")

            with st.expander("📋 Ver tabla completa de dividendos"):
                if df_todos is not None and not df_todos.empty:
                    df_show = df_todos.copy()
                    df_show["Ticker"] = df_show["Ticker"].str.replace(".BA", "", regex=False)
                    st.dataframe(
                        df_show.style.format({
                            "Dividend_Yield (%)": "{:.2f}%",
                            "Dividendo_Total":    "{:.4f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

            st.markdown(
                '<p class="warning-text">* Dividend Yield = dividendos anuales / precio actual. '
                'Datos de Yahoo Finance. Solo con fines educativos.</p>',
                unsafe_allow_html=True,
            )

    elif not div_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">💰</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado y hacé clic en<br>
                <strong style="color:#1a1a1a;">ESCANEAR DIVIDENDOS</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)



# ══════════════════════════════════════════════
# HELPER COMPARTIDO — TABS DE FUNDAMENTALS
# ══════════════════════════════════════════════

def _render_fundamentals_tab(
    tab_key: str,
    titulo: str,
    icono: str,
    descripcion: str,
    metrica_col: str,
    formato_fn,
    color_top: str = "#1a7a4a",
    color_bot: str = "#c0392b",
    nota: str = "",
):
    st.markdown(f"### {icono} {titulo}")
    st.markdown(descripcion)
    st.markdown("---")

    fc1, fc2, fc3 = st.columns([2, 1, 1])

    with fc1:
        fund_mercado = st.radio(
            "Mercado:",
            options=["NYSE / NASDAQ", "Mercado Argentino (ByMA)"],
            horizontal=True,
            key=f"{tab_key}_mercado",
        )

    with fc2:
        fund_vista = st.selectbox(
            "Vista:",
            options=["Tablas", "Gráfico de barras"],
            key=f"{tab_key}_vista",
        )

    with fc3:
        if fund_vista == "Gráfico de barras":
            fund_periodo_g = st.selectbox(
                "Período para el gráfico:",
                options=list(PERIODOS_FUND.keys()),
                key=f"{tab_key}_periodo_g",
            )
        else:
            fund_periodo_g = None

    st.markdown("<br>", unsafe_allow_html=True)
    fund_btn = st.button(f"🔬  ESCANEAR {titulo.upper()}", key=f"btn_{tab_key}")

    if fund_btn:
        mercado_key = "byma" if "ByMA" in fund_mercado else "nyse"
        with st.spinner(f"Descargando datos fundamentales... (puede tardar ~2 min)"):
            try:
                scanner_f = FundamentalsScanner(mercado=mercado_key)
                prog_f = st.progress(0, text="Analizando datos...")

                def _prog_f(val):
                    prog_f.progress(min(val, 1.0), text=f"Analizando... {int(val*100)}%")

                if tab_key == "per":
                    res_f = scanner_f.escanear_per(progress_callback=_prog_f)
                elif tab_key == "roe":
                    res_f = scanner_f.escanear_roe(progress_callback=_prog_f)
                else:
                    res_f = scanner_f.escanear_margen(progress_callback=_prog_f)

                prog_f.empty()
                st.session_state[f"{tab_key}_resultados"]  = res_f
                st.session_state[f"{tab_key}_mercado_lbl"] = fund_mercado
                st.session_state[f"{tab_key}_vista_val"]   = fund_vista
                st.session_state[f"{tab_key}_periodo_g"]   = fund_periodo_g

            except Exception as e:
                st.error(f"Error al escanear: {e}")

    if f"{tab_key}_resultados" in st.session_state:
        res_f     = st.session_state[f"{tab_key}_resultados"]
        vista_f   = st.session_state.get(f"{tab_key}_vista_val", "Tablas")
        periodo_g = st.session_state.get(f"{tab_key}_periodo_g", "12 meses")

        if vista_f == "Tablas":
            periodos_nombres = list(PERIODOS_FUND.keys())
            st.markdown(f'<div class="scanner-header" style="color:{color_top};">▲ Top 10 — {metrica_col} más alto</div>', unsafe_allow_html=True)
            cols_top = st.columns(4)
            for col_w, periodo in zip(cols_top, periodos_nombres):
                df_top = res_f.get(periodo, {}).get("top")
                with col_w:
                    st.markdown(f'<div class="scanner-period-label">⏱ {periodo}</div>', unsafe_allow_html=True)
                    if df_top is None or df_top.empty:
                        st.caption("Sin datos.")
                    else:
                        rows_html = ""
                        for i, row in df_top.iterrows():
                            ticker = str(row["Ticker"]).replace(".BA", "")
                            nombre = str(row.get("Nombre", ticker))[:22]
                            val    = row[metrica_col]
                            rows_html += f"""
                            <div class="rank-row">
                                <span class="rank-num">{i+1}</span>
                                <span class="rank-ticker">{ticker}</span>
                                <span style="font-size:0.72rem; color:#aaa; flex:2; padding:0 0.3rem; overflow:hidden; white-space:nowrap;">{nombre}</span>
                                <span style="font-family:'DM Sans',sans-serif; font-weight:500; color:{color_top};">{formato_fn(val)}</span>
                            </div>"""
                        st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="scanner-header" style="color:{color_bot};">▼ Top 10 — {metrica_col} más bajo</div>', unsafe_allow_html=True)
            cols_bot = st.columns(4)
            for col_w, periodo in zip(cols_bot, periodos_nombres):
                df_bot = res_f.get(periodo, {}).get("bottom")
                with col_w:
                    st.markdown(f'<div class="scanner-period-label">⏱ {periodo}</div>', unsafe_allow_html=True)
                    if df_bot is None or df_bot.empty:
                        st.caption("Sin datos.")
                    else:
                        rows_html = ""
                        for i, row in df_bot.iterrows():
                            ticker = str(row["Ticker"]).replace(".BA", "")
                            nombre = str(row.get("Nombre", ticker))[:22]
                            val    = row[metrica_col]
                            rows_html += f"""
                            <div class="rank-row">
                                <span class="rank-num">{i+1}</span>
                                <span class="rank-ticker">{ticker}</span>
                                <span style="font-size:0.72rem; color:#aaa; flex:2; padding:0 0.3rem; overflow:hidden; white-space:nowrap;">{nombre}</span>
                                <span style="font-family:'DM Sans',sans-serif; font-weight:500; color:{color_bot};">{formato_fn(val)}</span>
                            </div>"""
                        st.markdown(f'<div class="scanner-card">{rows_html}</div>', unsafe_allow_html=True)

        else:
            periodo_sel = periodo_g or "12 meses"
            df_top = res_f.get(periodo_sel, {}).get("top")
            df_bot = res_f.get(periodo_sel, {}).get("bottom")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                if df_top is not None and not df_top.empty:
                    df_p = df_top.copy()
                    df_p["Ticker"] = df_p["Ticker"].str.replace(".BA", "", regex=False)
                    colors = [f"rgba(26,122,74,{0.35 + 0.065*i})" for i in range(len(df_p))]
                    fig = go.Figure(go.Bar(
                        x=df_p[metrica_col], y=df_p["Ticker"],
                        orientation="h", marker_color=colors,
                        text=df_p[metrica_col].apply(formato_fn),
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"▲ {metrica_col} más alto — {periodo_sel}",
                        xaxis_title=metrica_col,
                        yaxis=dict(autorange="reversed"),
                        height=430, template="plotly_white",
                        margin=dict(l=10, r=80, t=40, b=30),
                    )
                    st.plotly_chart(fig, width="stretch")

            with col_g2:
                if df_bot is not None and not df_bot.empty:
                    df_p2 = df_bot.copy()
                    df_p2["Ticker"] = df_p2["Ticker"].str.replace(".BA", "", regex=False)
                    colors2 = [f"rgba(192,57,43,{0.35 + 0.065*i})" for i in range(len(df_p2))]
                    fig2 = go.Figure(go.Bar(
                        x=df_p2[metrica_col], y=df_p2["Ticker"],
                        orientation="h", marker_color=colors2,
                        text=df_p2[metrica_col].apply(formato_fn),
                        textposition="outside",
                    ))
                    fig2.update_layout(
                        title=f"▼ {metrica_col} más bajo — {periodo_sel}",
                        xaxis_title=metrica_col,
                        yaxis=dict(autorange="reversed"),
                        height=430, template="plotly_white",
                        margin=dict(l=10, r=80, t=40, b=30),
                    )
                    st.plotly_chart(fig2, width="stretch")

        with st.expander("📋 Ver tabla completa de todos los períodos"):
            tab_cols = st.tabs(list(PERIODOS_FUND.keys()))
            for tc, pn in zip(tab_cols, PERIODOS_FUND.keys()):
                with tc:
                    df_top = res_f.get(pn, {}).get("top", pd.DataFrame())
                    df_bot = res_f.get(pn, {}).get("bottom", pd.DataFrame())
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("▲ Mayores")
                        if not df_top.empty:
                            st.dataframe(df_top.style.format({metrica_col: formato_fn}), hide_index=True, width="stretch")
                    with c2:
                        st.caption("▼ Menores")
                        if not df_bot.empty:
                            st.dataframe(df_bot.style.format({metrica_col: formato_fn}), hide_index=True, width="stretch")

        if nota:
            st.markdown(f'<p class="warning-text">{nota}</p>', unsafe_allow_html=True)

    elif not fund_btn:
        st.markdown(f"""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">{icono}</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado y hacé clic en<br>
                <strong style="color:#1a1a1a;">ESCANEAR {titulo.upper()}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — RANKING PER
# ══════════════════════════════════════════════
with tab_per:
    _render_fundamentals_tab(
        tab_key      = "per",
        titulo       = "Ranking PER",
        icono        = "📊",
        descripcion  = "El **Price-to-Earnings Ratio (PER)** indica cuántas veces está pagando el mercado las ganancias anuales de la empresa. Un PER alto puede indicar expectativas de crecimiento; uno bajo, posible subvaluación.",
        metrica_col  = "PER",
        formato_fn   = lambda v: f"{v:.1f}x",
        color_top    = "#2563eb",
        color_bot    = "#7c3aed",
        nota         = "* PER = Precio / Ganancia por Acción (trailing). PER negativos (empresas con pérdidas) excluidos del ranking. Datos de Yahoo Finance. Solo con fines educativos.",
    )


# ══════════════════════════════════════════════
# TAB 6 — RANKING ROE
# ══════════════════════════════════════════════
with tab_roe:
    _render_fundamentals_tab(
        tab_key      = "roe",
        titulo       = "Ranking ROE",
        icono        = "💹",
        descripcion  = "El **Return on Equity (ROE)** mide la rentabilidad de una empresa en relación a su patrimonio neto. Un ROE alto indica que la empresa genera buenos retornos para sus accionistas.",
        metrica_col  = "ROE (%)",
        formato_fn   = lambda v: f"{v:.1f}%",
        color_top    = "#1a7a4a",
        color_bot    = "#c0392b",
        nota         = "* ROE = Utilidad Neta (anualizada) / Patrimonio Neto. Datos de Yahoo Finance. Solo con fines educativos.",
    )


# ══════════════════════════════════════════════
# TAB 7 — RANKING MÁRGENES
# ══════════════════════════════════════════════
with tab_margen:
    _render_fundamentals_tab(
        tab_key      = "margen",
        titulo       = "Ranking Márgenes",
        icono        = "📉",
        descripcion  = "El **Margen Neto** indica qué porcentaje de los ingresos se convierte en ganancia neta. Un margen alto refleja mayor eficiencia operativa y poder de fijación de precios.",
        metrica_col  = "Margen Neto (%)",
        formato_fn   = lambda v: f"{v:.1f}%",
        color_top    = "#1a7a4a",
        color_bot    = "#c0392b",
        nota         = "* Margen Neto = Utilidad Neta / Ingresos Totales. Datos de Yahoo Finance. Solo con fines educativos.",
    )


# ══════════════════════════════════════════════
# TAB 8 — SMART RANKING MULTIFACTOR
# ══════════════════════════════════════════════
with tab_mf:
    st.markdown("### 🏆 Smart Ranking — Modelo Multifactor")
    st.markdown(
        "Ranking cuantitativo que combina **Momentum**, **Valoración**, **Calidad** y **Dividendos** "
        "en un único **Smart Score** de 0 a 100. Metodología basada en *factor investing*."
    )
    st.markdown("---")

    with st.expander("📖 ¿Cómo funciona el modelo?"):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown("""
**Factores y pesos:**

| Factor | Componentes | Peso |
|--------|------------|------|
| 🚀 Momentum | Retorno 6m + 12m promedio | **40%** |
| 💲 Valoración | PER inverso (menor PER = mejor) | **20%** |
| ⭐ Calidad | ROA + ROE + Margen Neto | **30%** |
| 💰 Dividendos | Dividend Yield | **10%** |
""")
        with col_e2:
            st.markdown("""
**Metodología:**

1. Para cada factor se calcula un **percentil rank** de 0 a 100 dentro del universo de acciones.
2. Se ponderan los scores según los pesos definidos.
3. El **Smart Score final** es un número de 0 a 100 donde **100 = mejor combinación posible** en el universo analizado.
4. Para PER: menor PER = mayor score (se invierte el ranking).
5. Acciones sin datos en algún factor reciben un score neutro de 50 en ese factor.
""")

    st.markdown("<br>", unsafe_allow_html=True)

    mf_c1, mf_c2 = st.columns([3, 1])

    with mf_c1:
        mf_mercado = st.radio(
            "Mercado:",
            options=["NYSE / NASDAQ", "Mercado Argentino (ByMA)"],
            horizontal=True,
            key="mf_mercado",
        )

    with mf_c2:
        mf_vista = st.selectbox(
            "Vista adicional:",
            options=["Solo tabla", "Tabla + Gráfico radar"],
            key="mf_vista",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    mf_btn = st.button("🏆  CALCULAR SMART RANKING", key="btn_mf")

    if mf_btn:
        mercado_key = "byma" if "ByMA" in mf_mercado else "nyse"
        prog_mf = st.progress(0, text="Iniciando análisis multifactor...")

        def _prog_mf(val, msg="Procesando..."):
            prog_mf.progress(min(float(val), 1.0), text=msg)

        try:
            scanner_mf = MultifactorScanner(mercado=mercado_key)
            df_mf = scanner_mf.calcular(progress_callback=_prog_mf)
            prog_mf.empty()

            if df_mf is None or df_mf.empty:
                st.warning("No se pudieron calcular suficientes datos para el ranking.")
            else:
                st.session_state["mf_resultado"]    = df_mf
                st.session_state["mf_mercado_lbl"]  = mf_mercado
                st.session_state["mf_vista_val"]    = mf_vista

        except Exception as e:
            prog_mf.empty()
            st.error(f"Error al calcular el ranking: {e}")

    if "mf_resultado" in st.session_state:
        df_res   = st.session_state["mf_resultado"]
        lbl_mf   = st.session_state.get("mf_mercado_lbl", "")
        vista_mf = st.session_state.get("mf_vista_val", "Solo tabla")

        st.success(f"🏆 Smart Ranking — Top {len(df_res)} acciones · {lbl_mf}")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="scanner-header">📋 Tabla de Resultados</div>', unsafe_allow_html=True)

        df_tabla = df_res.copy()
        df_tabla.index.name = "#"
        df_tabla["Ticker"] = df_tabla["Ticker"].str.replace(".BA", "", regex=False)
        df_tabla["Nombre"] = df_tabla["Nombre"].str[:28]

        df_tabla = df_tabla.rename(columns={
            "smart_score":       "⭐ Smart Score",
            "score_momentum":    "🚀 Mom (40%)",
            "score_valoracion":  "💲 Val (20%)",
            "score_calidad":     "⭐ Cal (30%)",
            "score_dividendos":  "💰 Div (10%)",
            "mom_6m":            "Mom 6m %",
            "mom_12m":           "Mom 12m %",
            "PER":               "PER",
            "ROA_pct":           "ROA %",
            "ROE_pct":           "ROE %",
            "Margen_pct":        "Margen %",
            "DY_pct":            "Div Yield %",
        })

        score_cols = ["⭐ Smart Score", "🚀 Mom (40%)", "💲 Val (20%)", "⭐ Cal (30%)", "💰 Div (10%)"]

        def _color_score(val):
            if pd.isna(val):
                return ""
            if val >= 75:
                return "background-color: #d4edda; color: #1a7a4a; font-weight: bold;"
            elif val >= 50:
                return "background-color: #dbeafe; color: #1d4ed8; font-weight: bold;"
            elif val >= 25:
                return "background-color: #fef3c7; color: #d97706; font-weight: bold;"
            else:
                return "background-color: #fde8e8; color: #c0392b; font-weight: bold;"

        def _color_pct(val):
            if pd.isna(val):
                return ""
            return "color: #1a7a4a; font-weight:500;" if val >= 0 else "color: #c0392b; font-weight:500;"

        pct_cols = ["Mom 6m %", "Mom 12m %", "ROA %", "ROE %", "Margen %"]

        styled = (
            df_tabla.style
            .applymap(_color_score, subset=score_cols)
            .applymap(_color_pct,   subset=pct_cols)
            .format({
                "⭐ Smart Score": "{:.1f}",
                "🚀 Mom (40%)":  "{:.1f}",
                "💲 Val (20%)":  "{:.1f}",
                "⭐ Cal (30%)":  "{:.1f}",
                "💰 Div (10%)":  "{:.1f}",
                "Mom 6m %":      "{:+.1f}%",
                "Mom 12m %":     "{:+.1f}%",
                "PER":           "{:.1f}x",
                "ROA %":         "{:+.1f}%",
                "ROE %":         "{:+.1f}%",
                "Margen %":      "{:+.1f}%",
                "Div Yield %":   "{:.2f}%",
            }, na_rep="—")
            .set_properties(**{"text-align": "center"})
            .set_properties(subset=["Ticker", "Nombre"], **{"text-align": "left"})
        )

        st.dataframe(styled, width='stretch', height=720)

        if vista_mf == "Tabla + Gráfico radar":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scanner-header">📊 Visualización por Factor</div>', unsafe_allow_html=True)

            df_top10 = df_res.head(10).copy()
            df_top10["label"] = df_top10["Ticker"].str.replace(".BA", "", regex=False)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                colors_ss = [
                    "#1a7a4a" if s >= 75 else "#2563eb" if s >= 50 else "#f59e0b" if s >= 25 else "#c0392b"
                    for s in df_top10["smart_score"]
                ]
                fig_ss = go.Figure(go.Bar(
                    x=df_top10["smart_score"],
                    y=df_top10["label"],
                    orientation="h",
                    marker_color=colors_ss,
                    text=df_top10["smart_score"].apply(lambda v: f"{v:.1f}"),
                    textposition="outside",
                ))
                fig_ss.update_layout(
                    title="⭐ Smart Score — Top 10",
                    xaxis=dict(range=[0, 105], title="Smart Score"),
                    yaxis=dict(autorange="reversed"),
                    height=400, template="plotly_white",
                    margin=dict(l=10, r=60, t=40, b=30),
                )
                st.plotly_chart(fig_ss, width="stretch")

            with col_g2:
                fig_stack = go.Figure()
                factor_config = [
                    ("score_momentum",   "🚀 Momentum (40%)",   "rgba(37,99,235,0.75)"),
                    ("score_valoracion", "💲 Valoración (20%)",  "rgba(124,58,237,0.75)"),
                    ("score_calidad",    "⭐ Calidad (30%)",     "rgba(26,122,74,0.75)"),
                    ("score_dividendos", "💰 Dividendos (10%)", "rgba(245,158,11,0.75)"),
                ]
                pesos_list = [PESOS["momentum"], PESOS["valoracion"], PESOS["calidad"], PESOS["dividendos"]]

                for (col_name, label, color), peso in zip(factor_config, pesos_list):
                    vals = (df_top10[col_name] * peso).round(1)
                    fig_stack.add_trace(go.Bar(
                        name=label,
                        x=vals,
                        y=df_top10["label"],
                        orientation="h",
                        marker_color=color,
                        text=vals.apply(lambda v: f"{v:.0f}"),
                        textposition="inside",
                    ))

                fig_stack.update_layout(
                    title="📊 Composición del Smart Score",
                    barmode="stack",
                    xaxis=dict(range=[0, 105], title="Score ponderado"),
                    yaxis=dict(autorange="reversed"),
                    height=400, template="plotly_white",
                    margin=dict(l=10, r=20, t=40, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
                )
                st.plotly_chart(fig_stack, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 **Interpretación del Smart Score:** "
            "Score ≥ 75 🟢 Muy favorable · 50–74 🔵 Favorable · 25–49 🟡 Moderado · < 25 🔴 Bajo. "
            "El score es relativo al universo analizado, no absoluto."
        )
        st.markdown(
            '<p class="warning-text">* Smart Score basado en percentil rank dentro del universo de acciones analizado. '
            'No constituye recomendación de inversión. Datos de Yahoo Finance. Solo con fines educativos.</p>',
            unsafe_allow_html=True,
        )

    elif not mf_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">🏆</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado y hacé clic en<br>
                <strong style="color:#1a1a1a;">CALCULAR SMART RANKING</strong>
            </p>
            <p style="font-size:0.85rem; color:#bbb; margin-top:0.75rem;">
                El análisis puede tardar ~2 minutos por la cantidad de datos a descargar.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 9 — CONTRARIAN RANKING
# ══════════════════════════════════════════════
with tab_contrarian:
    st.markdown("### 🎯 Contrarian Ranking — Caídas con Mejores Fundamentos")
    st.markdown(
        "Identifica las acciones con **mayor caída de precio** en el período seleccionado "
        "y las filtra por **calidad fundamental**. La idea: separar las caídas por pánico de mercado "
        "de las caídas por deterioro real del negocio."
    )
    st.markdown("---")

    with st.expander("📖 ¿Cómo funciona el Contrarian Ranking?"):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown("""
**Metodología en 3 pasos:**

1. 📉 Se toman las **40 acciones con mayor caída** de precio en el período elegido.
2. 🔬 A ese universo se le aplica el mismo **scoring multifactor** del Smart Ranking.
3. 🏆 Se presentan las **Top 20** con mejor combinación de fundamentos.

**La hipótesis:** si una empresa cayó mucho pero mantiene buenos fundamentos (calidad, valoración razonable, momentum relativo), puede ser una oportunidad de rebote.
""")
        with col_e2:
            st.markdown("""
**Factores y pesos:**

| Factor | Componentes | Peso |
|--------|------------|------|
| 🚀 Momentum | Retorno 6m + 12m promedio | **40%** |
| 💲 Valoración | PER inverso (menor PER = mejor) | **20%** |
| ⭐ Calidad | ROA + ROE + Margen Neto | **30%** |
| 💰 Dividendos | Dividend Yield | **10%** |

> El Smart Score mide qué tan bueno es el negocio *dentro del grupo de caídas*, no en el mercado general.
""")

    st.markdown("<br>", unsafe_allow_html=True)

    ct_c1, ct_c2, ct_c3 = st.columns([2, 2, 1])

    with ct_c1:
        ct_mercado = st.radio(
            "Mercado:",
            options=["NYSE / NASDAQ", "Mercado Argentino (ByMA)"],
            horizontal=True,
            key="ct_mercado",
        )

    with ct_c2:
        ct_periodo = st.radio(
            "Período de caída a analizar:",
            options=list(PERIODOS_CONTRARIAN.keys()),
            horizontal=True,
            key="ct_periodo",
        )

    with ct_c3:
        ct_vista = st.selectbox(
            "Vista:",
            options=["Tabla + Gráficos", "Solo tabla"],
            key="ct_vista",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    ct_btn = st.button("🎯  CALCULAR CONTRARIAN RANKING", key="btn_ct")

    if ct_btn:
        mercado_key = "byma" if "ByMA" in ct_mercado else "nyse"
        dias_sel    = PERIODOS_CONTRARIAN[ct_periodo]
        prog_ct     = st.progress(0, text="Iniciando análisis contrarian...")

        def _prog_ct(val, msg="Procesando..."):
            prog_ct.progress(min(float(val), 1.0), text=msg)

        try:
            scanner_ct = ContrarianScanner(mercado=mercado_key)
            df_ct      = scanner_ct.calcular(dias=dias_sel, progress_callback=_prog_ct)
            prog_ct.empty()

            if df_ct is None or df_ct.empty:
                st.warning("No se pudieron calcular suficientes datos para el ranking.")
            else:
                st.session_state["ct_resultado"]   = df_ct
                st.session_state["ct_mercado_lbl"] = ct_mercado
                st.session_state["ct_periodo_lbl"] = ct_periodo
                st.session_state["ct_vista_val"]   = ct_vista

        except Exception as e:
            prog_ct.empty()
            st.error(f"Error al calcular el ranking: {e}")

    if "ct_resultado" in st.session_state:
        df_res   = st.session_state["ct_resultado"]
        lbl_ct   = st.session_state.get("ct_mercado_lbl", "")
        prd_ct   = st.session_state.get("ct_periodo_lbl", "")
        vista_ct = st.session_state.get("ct_vista_val", "Tabla + Gráficos")

        st.success(f"🎯 Contrarian Ranking — Top {len(df_res)} · {lbl_ct} · Caídas {prd_ct}")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="scanner-header">📋 Resultados — Caídas con Mejores Fundamentos</div>',
                    unsafe_allow_html=True)

        df_tabla = df_res.copy()
        df_tabla.index.name = "#"
        df_tabla["Ticker"] = df_tabla["Ticker"].str.replace(".BA", "", regex=False)
        df_tabla["Nombre"] = df_tabla["Nombre"].str[:28]

        df_tabla = df_tabla.rename(columns={
            "caida_periodo":     f"📉 Caída {prd_ct}",
            "smart_score":       "⭐ Smart Score",
            "score_momentum":    "🚀 Mom (40%)",
            "score_valoracion":  "💲 Val (20%)",
            "score_calidad":     "⭐ Cal (30%)",
            "score_dividendos":  "💰 Div (10%)",
            "mom_6m":            "Mom 6m %",
            "mom_12m":           "Mom 12m %",
            "PER":               "PER",
            "ROA_pct":           "ROA %",
            "ROE_pct":           "ROE %",
            "Margen_pct":        "Margen %",
            "DY_pct":            "Div Yield %",
        })

        caida_col   = f"📉 Caída {prd_ct}"
        score_cols  = ["⭐ Smart Score", "🚀 Mom (40%)", "💲 Val (20%)", "⭐ Cal (30%)", "💰 Div (10%)"]

        def _color_score(val):
            if pd.isna(val):
                return ""
            if val >= 75:
                return "background-color: #d4edda; color: #1a7a4a; font-weight: bold;"
            elif val >= 50:
                return "background-color: #dbeafe; color: #1d4ed8; font-weight: bold;"
            elif val >= 25:
                return "background-color: #fef3c7; color: #d97706; font-weight: bold;"
            else:
                return "background-color: #fde8e8; color: #c0392b; font-weight: bold;"

        def _color_caida(val):
            if pd.isna(val):
                return ""
            return "color: #c0392b; font-weight: bold;"

        def _color_pct(val):
            if pd.isna(val):
                return ""
            return "color: #1a7a4a; font-weight:500;" if val >= 0 else "color: #c0392b; font-weight:500;"

        pct_cols = ["Mom 6m %", "Mom 12m %", "ROA %", "ROE %", "Margen %"]

        format_dict = {
            caida_col:       "{:+.1f}%",
            "⭐ Smart Score": "{:.1f}",
            "🚀 Mom (40%)":  "{:.1f}",
            "💲 Val (20%)":  "{:.1f}",
            "⭐ Cal (30%)":  "{:.1f}",
            "💰 Div (10%)":  "{:.1f}",
            "Mom 6m %":      "{:+.1f}%",
            "Mom 12m %":     "{:+.1f}%",
            "ROA %":         "{:+.1f}%",
            "ROE %":         "{:+.1f}%",
            "Margen %":      "{:+.1f}%",
            "Div Yield %":   "{:.2f}%",
        }
        if "PER" in df_tabla.columns:
            format_dict["PER"] = "{:.1f}x"

        styled = (
            df_tabla.style
            .applymap(_color_score, subset=score_cols)
            .applymap(_color_caida, subset=[caida_col])
            .applymap(_color_pct,   subset=pct_cols)
            .format(format_dict, na_rep="—")
            .set_properties(**{"text-align": "center"})
            .set_properties(subset=["Ticker", "Nombre"], **{"text-align": "left"})
        )

        st.dataframe(styled, width='stretch', height=720)

        if vista_ct == "Tabla + Gráficos":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="scanner-header">📊 Visualización</div>', unsafe_allow_html=True)

            df_top10 = df_res.head(10).copy()
            df_top10["label"] = df_top10["Ticker"].str.replace(".BA", "", regex=False)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                fig_scatter = go.Figure()

                colors_scatter = []
                for _, row in df_res.iterrows():
                    if row["smart_score"] >= 75:
                        colors_scatter.append("#1a7a4a")
                    elif row["smart_score"] >= 50:
                        colors_scatter.append("#2563eb")
                    elif row["smart_score"] >= 25:
                        colors_scatter.append("#f59e0b")
                    else:
                        colors_scatter.append("#c0392b")

                fig_scatter.add_trace(go.Scatter(
                    x=df_res["caida_periodo"],
                    y=df_res["smart_score"],
                    mode="markers+text",
                    marker=dict(size=12, color=colors_scatter, opacity=0.85,
                                line=dict(color="white", width=1)),
                    text=df_res["Ticker"].str.replace(".BA", "", regex=False),
                    textposition="top center",
                    textfont=dict(size=9),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Caída: %{x:.1f}%<br>"
                        "Smart Score: %{y:.1f}<br>"
                        "<extra></extra>"
                    ),
                ))

                fig_scatter.add_vline(
                    x=df_res["caida_periodo"].median(),
                    line_dash="dot", line_color="#ccc",
                    annotation_text="Mediana caída", annotation_position="top right",
                    annotation_font_size=9,
                )
                fig_scatter.add_hline(
                    y=50, line_dash="dot", line_color="#ccc",
                    annotation_text="Score 50", annotation_position="right",
                    annotation_font_size=9,
                )

                fig_scatter.update_layout(
                    title=f"📍 Caída vs Smart Score — {prd_ct}",
                    xaxis_title=f"Caída de precio {prd_ct} (%)",
                    yaxis_title="Smart Score (0–100)",
                    height=430, template="plotly_white",
                    margin=dict(l=10, r=20, t=50, b=40),
                    xaxis=dict(ticksuffix="%"),
                )
                st.plotly_chart(fig_scatter, width="stretch")
                st.caption("💡 Zona ideal (esquina superior izquierda): mayor caída + mayor Smart Score.")

            with col_g2:
                colors_bar = [
                    "#1a7a4a" if s >= 75 else "#2563eb" if s >= 50 else "#f59e0b" if s >= 25 else "#c0392b"
                    for s in df_top10["smart_score"]
                ]
                caida_col_raw = "caida_periodo"
                custom_text = [
                    f"{ss:.0f} pts (caída: {cd:+.1f}%)"
                    for ss, cd in zip(df_top10["smart_score"], df_top10[caida_col_raw])
                ]
                fig_bar = go.Figure(go.Bar(
                    x=df_top10["smart_score"],
                    y=df_top10["label"],
                    orientation="h",
                    marker_color=colors_bar,
                    text=custom_text,
                    textposition="outside",
                    textfont=dict(size=10),
                ))
                fig_bar.update_layout(
                    title="🏆 Top 10 — Smart Score + Caída",
                    xaxis=dict(range=[0, 115], title="Smart Score"),
                    yaxis=dict(autorange="reversed"),
                    height=430, template="plotly_white",
                    margin=dict(l=10, r=120, t=50, b=30),
                )
                st.plotly_chart(fig_bar, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)
            fig_stack = go.Figure()
            factor_config = [
                ("score_momentum",   "🚀 Momentum (40%)",   "rgba(37,99,235,0.80)"),
                ("score_valoracion", "💲 Valoración (20%)",  "rgba(124,58,237,0.80)"),
                ("score_calidad",    "⭐ Calidad (30%)",     "rgba(26,122,74,0.80)"),
                ("score_dividendos", "💰 Dividendos (10%)", "rgba(245,158,11,0.80)"),
            ]
            from contrarian_model import PESOS_C
            pesos_list = [PESOS_C["momentum"], PESOS_C["valoracion"], PESOS_C["calidad"], PESOS_C["dividendos"]]

            for (col_name, label, color), peso in zip(factor_config, pesos_list):
                vals = (df_top10[col_name] * peso).round(1)
                fig_stack.add_trace(go.Bar(
                    name=label,
                    x=vals,
                    y=df_top10["label"],
                    orientation="h",
                    marker_color=color,
                    text=vals.apply(lambda v: f"{v:.0f}"),
                    textposition="inside",
                    textfont=dict(size=10, color="white"),
                ))

            fig_stack.update_layout(
                title="📊 Composición del Smart Score — Top 10",
                barmode="stack",
                xaxis=dict(range=[0, 105], title="Score ponderado"),
                yaxis=dict(autorange="reversed"),
                height=420, template="plotly_white",
                margin=dict(l=10, r=20, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
            )
            st.plotly_chart(fig_stack, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 **Cómo interpretar el ranking:** "
            "Buscá acciones con **caída grande** (columna roja) y **Smart Score alto** (verde). "
            "Eso indica que el precio cayó pero los fundamentos del negocio siguen siendo sólidos. "
            "Score ≥ 75 🟢 Muy favorables · 50–74 🔵 Buenos · 25–49 🟡 Moderados · < 25 🔴 Débiles."
        )
        st.markdown(
            '<p class="warning-text">* Este ranking identifica caídas con buenos fundamentos, no garantiza rebote. '
            'Datos de Yahoo Finance. Solo con fines educativos. No constituye recomendación de inversión.</p>',
            unsafe_allow_html=True,
        )

    elif not ct_btn:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 2rem; color:#bbb;">
            <p style="font-size:4rem; margin:0;">🎯</p>
            <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#aaa; margin-top:1rem;">
                Seleccioná el mercado y el período de caída,<br>
                luego hacé clic en <strong style="color:#1a1a1a;">CALCULAR CONTRARIAN RANKING</strong>
            </p>
            <p style="font-size:0.85rem; color:#bbb; margin-top:0.75rem;">
                El análisis puede tardar ~2 minutos por la cantidad de datos a descargar.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 10 — ANÁLISIS DE MERCADO
# ══════════════════════════════════════════════
with tab_mercado:
    st.markdown("### 🧭 Análisis de Mercado")
    st.caption(
        "Top 20 activos por capitalización de mercado, actualizado a la fecha de hoy. "
        "Se recalcula 1 vez por día para no sobrecargar Yahoo Finance."
    )

    mercado_sel = st.radio(
        "Mercado", options=list(MERCADOS_DISPONIBLES.keys()),
        format_func=lambda k: MERCADOS_DISPONIBLES[k],
        horizontal=True, key="mercado_analisis",
    )

    with st.spinner("Cargando datos de mercado (puede tardar la primera vez del día)..."):
        analyzer_mercado = cargar_market_analyzer(periodo="3y")
        top20_df = analyzer_mercado.top20(mercado_sel)

    if top20_df.empty:
        st.warning("No se pudo armar el Top 20 (revisar disponibilidad de market cap para este mercado).")
    else:
        if mercado_sel == "byma":
            st.info(
                "💡 El ranking y el gráfico de rendimiento acumulado de Merval usan los **ADRs en USD** "
                "de cada empresa (cuando existen) para poder compararlos contra el S&P 500 en la misma "
                "moneda — comparar retornos en pesos contra el S&P 500 en dólares no sería válido."
            )

        # ── Mapa Riesgo vs Retorno ──
        st.markdown(f"#### Mapa Riesgo vs Retorno — {MERCADOS_DISPONIBLES[mercado_sel]}")
        x_mediana = top20_df["volatilidad_pct"].median()

        def _clasificar(row):
            if row["volatilidad_pct"] < x_mediana and row["retorno_pct"] > 0:
                return "Alto retorno / Bajo riesgo"
            elif row["volatilidad_pct"] >= x_mediana and row["retorno_pct"] <= 0:
                return "Bajo retorno / Alto riesgo"
            return "Mixto"

        top20_df = top20_df.copy()
        top20_df["cuadrante"] = top20_df.apply(_clasificar, axis=1)
        colores_cuadrante = {
            "Alto retorno / Bajo riesgo": "#2ca02c",
            "Bajo retorno / Alto riesgo": "#d62728",
            "Mixto": "#1f77b4",
        }

        fig_riesgo = go.Figure()
        for cuadrante, color in colores_cuadrante.items():
            sub = top20_df[top20_df["cuadrante"] == cuadrante]
            if sub.empty:
                continue
            fig_riesgo.add_trace(go.Scatter(
                x=sub["volatilidad_pct"], y=sub["retorno_pct"],
                mode="markers+text",
                text=[t.replace(".BA", "") for t in sub.index],
                textposition="top center",
                marker=dict(size=14, color=color, line=dict(width=1, color="white")),
                name=cuadrante,
                hovertemplate="<b>%{text}</b><br>Volatilidad: %{x:.1f}%<br>Retorno: %{y:.1f}%<extra></extra>",
            ))
        fig_riesgo.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_riesgo.add_vline(x=x_mediana, line_dash="dot", line_color="gray")
        fig_riesgo.update_layout(
            xaxis_title="Volatilidad anualizada (%) — Riesgo →",
            yaxis_title="Retorno anualizado (%) — Ganancia ↑",
            height=550, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_riesgo, width="stretch")

        # ── Rendimiento acumulado vs SPY ──
        st.markdown(f"#### Rendimiento Acumulado vs. S&P 500 — {MERCADOS_DISPONIBLES[mercado_sel]}")
        serie_acum = analyzer_mercado.serie_acumulada(mercado_sel, meses=36)

        fig_acum = go.Figure()
        colores_lineas = px.colors.qualitative.Alphabet
        columnas_activos = [c for c in serie_acum.columns if c != "SPY"]
        for i, col in enumerate(columnas_activos):
            fig_acum.add_trace(go.Scatter(
                x=serie_acum.index, y=serie_acum[col], mode="lines",
                name=col.replace(".BA", ""),
                line=dict(width=1.4, color=colores_lineas[i % len(colores_lineas)]),
            ))
        if "SPY" in serie_acum.columns:
            fig_acum.add_trace(go.Scatter(
                x=serie_acum.index, y=serie_acum["SPY"], mode="lines",
                name="SPY (S&P 500)", line=dict(width=3, color="black", dash="dash"),
            ))
        fig_acum.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_acum.update_layout(
            yaxis_title="Índice (base 100)", height=600, template="plotly_white",
            hovermode="x unified",
            updatemenus=[dict(
                type="buttons", direction="right", x=0.0, y=1.1, xanchor="left",
                buttons=[
                    dict(label="Escala lineal", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="Escala log", method="relayout", args=[{"yaxis.type": "log"}]),
                ],
            )],
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(buttons=[
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1a", step="year", stepmode="backward"),
                    dict(step="all", label="Todo"),
                ]),
            ),
        )
        st.plotly_chart(fig_acum, width="stretch")

        # ── Ranking ──
        st.markdown(f"#### Ranking — {MERCADOS_DISPONIBLES[mercado_sel]}")
        ranking_df = analyzer_mercado.ranking(mercado_sel, meses=36)
        if ranking_df.empty:
            st.caption("Sin datos suficientes para el ranking en este mercado.")
        else:
            cols_mostrar = ["retorno_pct", "volatilidad_pct", "sharpe_simple",
                             "retorno_acumulado_pct", "alpha_vs_spy_pct", "score_final"]
            ranking_mostrar = ranking_df[cols_mostrar].rename(columns={
                "retorno_pct": "Retorno Anual (%)",
                "volatilidad_pct": "Volatilidad Anual (%)",
                "sharpe_simple": "Sharpe",
                "retorno_acumulado_pct": "Retorno Acumulado 3a (%)",
                "alpha_vs_spy_pct": "Alpha vs. SPY (%)",
                "score_final": "Score (0-100)",
            }).round(1)
            ranking_mostrar.index.name = "Ticker"
            st.dataframe(ranking_mostrar, width="stretch")

        st.markdown(
            '<p class="warning-text">* Ranking basado en performance histórica (Sharpe + retorno acumulado, '
            'pesos iguales). No es una predicción ni una recomendación de inversión a futuro. '
            'Datos de Yahoo Finance, solo con fines educativos.</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# TAB 11 — SIMULADOR PERSONALIZADO
# ══════════════════════════════════════════════
with tab_simulador:
    st.markdown("### 🧪 Simulador Personalizado")
    st.caption(
        "Elegí tus propios activos y pesos, y mirá cómo hubiese rendido ese portafolio "
        "comparado contra el S&P 500. Los activos sugeridos son el Top 20 de cada mercado, "
        "pero podés elegir cualquier activo del universo completo."
    )

    with st.spinner("Cargando sugerencias..."):
        analyzer_sim = cargar_market_analyzer(periodo="3y")
        top20_byma_sim = analyzer_sim.top20("byma").index.tolist()
        top20_nyse_sim = analyzer_sim.top20("nyse").index.tolist()

    sugeridos = top20_nyse_sim + top20_byma_sim
    opciones_completas = sorted(set(UNIVERSO))

    col_sel, col_conf = st.columns([2, 1])

    with col_sel:
        mostrar_todo = st.checkbox("Mostrar el universo completo (no solo el Top 20 sugerido)", value=False)
        opciones_multiselect = opciones_completas if mostrar_todo else sorted(set(sugeridos))

        seleccionados_sim = st.multiselect(
            "Activos del portafolio",
            options=opciones_multiselect,
            default=[t for t in sugeridos[:3]],
            format_func=lambda t: t.replace(".BA", "") + (" (ByMA)" if t.endswith(".BA") else ""),
            key="sim_seleccion",
        )

    with col_conf:
        periodo_sim = st.selectbox(
            "Período de la simulación", options=["6mo", "1y", "2y", "3y", "5y"],
            index=2, key="sim_periodo",
        )

    pesos_sim = {}
    if seleccionados_sim:
        st.markdown("**Pesos del portafolio (%)**")
        peso_default_sim = int(100 / len(seleccionados_sim))
        total_sim = 0
        cols_pesos = st.columns(min(len(seleccionados_sim), 4))
        for i, ticker in enumerate(seleccionados_sim):
            with cols_pesos[i % len(cols_pesos)]:
                p = st.slider(ticker.replace(".BA", ""), 0, 100, peso_default_sim, 5, key=f"sim_peso_{ticker}")
                pesos_sim[ticker] = p
                total_sim += p

        if total_sim != 100:
            st.warning(f"Los pesos suman {total_sim}%. Se normalizarán automáticamente a 100%.")
        else:
            st.success("✓ Los pesos suman 100%")

    simular_btn = st.button("▶  SIMULAR PORTAFOLIO", key="btn_simular")

    if simular_btn:
        if not seleccionados_sim:
            st.warning("Seleccioná al menos un activo para continuar.")
        else:
            with st.spinner("Descargando datos y calculando..."):
                try:
                    tickers_con_spy = list(dict.fromkeys(seleccionados_sim + ["SPY"]))
                    analyzer_sim_port = PortfolioAnalyzer(tickers_con_spy, periodo_sim)
                    pesos_dec_sim = {t: p / 100 for t, p in pesos_sim.items()}
                    resultado_sim = analyzer_sim_port.analizar_portafolio(pesos_dec_sim)

                    spy_evolucion = (
                        analyzer_sim_port.precios["SPY"] / analyzer_sim_port.precios["SPY"].iloc[0]
                    ) * 100

                    st.session_state.resultado_sim = resultado_sim
                    st.session_state.spy_evolucion_sim = spy_evolucion
                except Exception as e:
                    st.error(f"Error al descargar datos: {e}")
                    st.stop()

    if "resultado_sim" in st.session_state:
        r_sim = st.session_state.resultado_sim
        spy_ev = st.session_state.spy_evolucion_sim

        col1, col2, col3, col4 = st.columns(4)
        ret_color = "positive" if r_sim["retorno_anual"] >= 0 else "negative"
        draw_color = "negative" if r_sim["max_drawdown"] < -15 else "neutral"

        with col1:
            st.markdown(f"""<div class="metric-card"><p class="metric-label">Retorno Anual Est.</p>
                <p class="metric-value {ret_color}">{r_sim['retorno_anual']:+.1f}%</p></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><p class="metric-label">Volatilidad Anual</p>
                <p class="metric-value neutral">{r_sim['volatilidad']:.1f}%</p></div>""",
                unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><p class="metric-label">Sharpe Ratio</p>
                <p class="metric-value neutral">{r_sim['sharpe']:.2f}</p></div>""",
                unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><p class="metric-label">Máx. Drawdown</p>
                <p class="metric-value {draw_color}">{r_sim['max_drawdown']:.1f}%</p></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=r_sim["fechas"], y=r_sim["evolucion"], mode="lines",
            name="Mi Portafolio", line=dict(width=3, color="#1a7a4a"),
        ))
        fig_sim.add_trace(go.Scatter(
            x=spy_ev.index, y=spy_ev.values, mode="lines",
            name="SPY (S&P 500)", line=dict(width=2.5, color="black", dash="dash"),
        ))
        fig_sim.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_sim.update_layout(
            title="Mi Portafolio vs. S&P 500 (base 100)",
            yaxis_title="Índice (base 100)", height=550, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sim, width="stretch")

        alpha_total = float(r_sim["evolucion"].iloc[-1] - spy_ev.reindex(r_sim["evolucion"].index).iloc[-1])
        if alpha_total >= 0:
            st.success(f"Tu portafolio le sacó **+{alpha_total:.1f} puntos** al S&P 500 en este período.")
        else:
            st.warning(f"Tu portafolio quedó **{alpha_total:.1f} puntos** por debajo del S&P 500 en este período.")

    st.markdown(
        '<p class="warning-text">* Simulación histórica ("qué hubiese pasado si"), no una proyección '
        'a futuro. Datos de Yahoo Finance, solo con fines educativos.</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#aaa; font-size:0.75rem;">
    Simulador de Portafolio · Datos: Yahoo Finance · Solo con fines educativos
</p>
""", unsafe_allow_html=True)
