"""
Market Analysis Model
======================
Análisis de mercado: mapa riesgo-retorno, rendimiento acumulado vs. SPY,
y ranking combinado (Sharpe + Alpha), por separado para ByMA y NYSE/NASDAQ.

Consolida la lógica validada en la investigación previa (Colab):
  - Manejo correcto de calendarios ByMA/NYSE: forward-fill, NUNCA dropna()
    por fila sobre el panel mezclado (eso borraba casi todas las fechas).
  - Rankings SIEMPRE separados por mercado — no comparar retornos en ARS
    contra retornos en USD directamente.
  - ADRs en USD para poder comparar Merval contra SPY en una moneda común.
  - Score = pesos iguales entre Sharpe simple y retorno acumulado (percentile
    rank). No se usa optimización de pesos: la investigación previa (Monte
    Carlo + walk-forward) mostró que no generaliza mejor que 1/N con este
    universo y horizonte de datos disponible.

Todas las descargas son relativas a "hoy" (yfinance interpreta `period`
desde la fecha de ejecución) — no hay fechas fijas en este módulo.

Autor: Luis / Investigación cuantitativa
Versión: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# ─────────────────────────────────────────────────────────────────
# UNIVERSO Y MAPEOS
# ─────────────────────────────────────────────────────────────────

UNIVERSO = sorted(set(BYMA_TICKERS + NYSE_TICKERS))


def mercado_de(ticker: str) -> str:
    return "byma" if ticker.endswith(".BA") else "nyse"


MERCADO_MAP = {t: mercado_de(t) for t in UNIVERSO}

# ADRs en USD para los activos de ByMA más líquidos. Permiten comparar
# contra SPY sin mezclar el efecto de la devaluación del peso.
MAPA_ADR = {
    "GGAL.BA": "GGAL", "YPFD.BA": "YPF", "PAMP.BA": "PAM", "BMA.BA": "BMA",
    "BBAR.BA": "BBAR", "CRES.BA": "CRESY", "IRSA.BA": "IRS", "TGSU2.BA": "TGS",
    "SUPV.BA": "SUPV", "LOMA.BA": "LOMA", "EDN.BA": "EDN", "CEPU.BA": "CEPU",
    "CVH.BA": "CVH", "TECO2.BA": "TEO",
}

MERCADOS_DISPONIBLES = {"byma": "Merval (Argentina)", "nyse": "NYSE/NASDAQ (USA)"}


# ─────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────

def descargar_universo(periodo: str = "3y") -> pd.DataFrame:
    """
    Descarga precios del universo completo + SPY + los ADRs conocidos.
    `periodo` es relativo a HOY (ej. '3y' = últimos 3 años desde la fecha
    de ejecución), así que cada corrida de la app usa datos actualizados.

    Aplica forward-fill para resolver el desalineamiento de calendarios
    entre ByMA y NYSE. NO usar dropna() por fila sobre este panel mixto:
    con dos calendarios distintos, casi ningún día tiene los ~350 tickers
    completos a la vez, y un dropna() de fila borraría casi todo el panel.
    """
    tickers_adr = list(MAPA_ADR.values())
    tickers_totales = sorted(set(UNIVERSO + ["SPY"] + tickers_adr))

    data = yf.download(tickers_totales, period=periodo, auto_adjust=True,
                        progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        precios = data["Close"]
    else:
        precios = data[["Close"]].rename(columns={"Close": tickers_totales[0]})

    precios = precios.dropna(axis=1, how="all")
    precios = precios.ffill()
    return precios


def obtener_market_caps(tickers: list, progress_callback=None) -> pd.Series:
    """Market cap actual por ticker (1 llamada por ticker). Cachear en la app."""
    caps = {}
    total = len(tickers)
    for i, t in enumerate(tickers):
        try:
            caps[t] = yf.Ticker(t).info.get("marketCap", None)
        except Exception:
            caps[t] = None
        if progress_callback and (i + 1) % 25 == 0:
            progress_callback((i + 1) / total)
    return pd.Series(caps, name="market_cap")


# ─────────────────────────────────────────────────────────────────
# MÉTRICAS DE RIESGO-RETORNO Y RENDIMIENTO ACUMULADO
# ─────────────────────────────────────────────────────────────────

def calcular_riesgo_retorno(precios: pd.DataFrame, tickers: list, meses: int = 12) -> pd.DataFrame:
    """Retorno y volatilidad anualizados sobre los últimos `meses`."""
    dias = int(meses * 21)
    cols = [t for t in tickers if t in precios.columns]
    ventana = precios[cols].iloc[-dias:]
    retornos = ventana.pct_change()

    ret_anual = retornos.mean() * 252 * 100
    vol_anual = retornos.std() * np.sqrt(252) * 100
    return pd.DataFrame({"retorno_pct": ret_anual, "volatilidad_pct": vol_anual})


def calcular_retorno_acumulado(precios: pd.DataFrame, tickers: list, meses: int = 36) -> pd.Series:
    """Retorno total (%) acumulado en la ventana de `meses`, base 100 al inicio."""
    dias = int(meses * 21)
    cols = [t for t in tickers if t in precios.columns]
    ventana = precios[cols].iloc[-dias:].dropna(how="all")
    if ventana.empty:
        return pd.Series(dtype=float)
    base = ventana.bfill().iloc[0]
    return (ventana.iloc[-1] / base - 1) * 100


def serie_base100(precios: pd.DataFrame, tickers: list, meses: int = 36) -> pd.DataFrame:
    """Serie de evolución normalizada a base 100, para graficar en el tiempo."""
    dias = int(meses * 21)
    cols = [t for t in tickers if t in precios.columns]
    sub = precios[cols].iloc[-dias:].dropna(axis=1, how="all")
    sub = sub.dropna(how="all")
    return (sub / sub.bfill().iloc[0]) * 100


# ─────────────────────────────────────────────────────────────────
# SELECCIÓN TOP N POR MERCADO
# ─────────────────────────────────────────────────────────────────

def seleccionar_top_n(precios: pd.DataFrame, market_caps: pd.Series, mercado: str,
                       n: int = 20, meses_riesgo: int = 12, criterio: str = "market_cap") -> pd.DataFrame:
    tickers_mercado = [t for t, m in MERCADO_MAP.items() if m == mercado]
    riesgo_retorno = calcular_riesgo_retorno(precios, tickers_mercado, meses_riesgo)
    df = riesgo_retorno.dropna(subset=["retorno_pct", "volatilidad_pct"]).copy()
    df["market_cap"] = market_caps.reindex(df.index)

    if criterio == "market_cap":
        df = df.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False)
    elif criterio == "sharpe_abs":
        df["sharpe"] = df["retorno_pct"] / df["volatilidad_pct"]
        df = df.reindex(df["sharpe"].abs().sort_values(ascending=False).index)
    # criterio == "todos": sin filtrar/ordenar adicional (útil si el mercado tiene pocos tickers)

    return df.head(n)


# ─────────────────────────────────────────────────────────────────
# RANKING: Sharpe simple + Alpha vs. SPY, pesos iguales
# ─────────────────────────────────────────────────────────────────

def construir_ranking(tickers: list, precios: pd.DataFrame, meses: int = 36,
                       usar_adr: bool = False) -> pd.DataFrame:
    """
    tickers: lista de tickers "nativos" (para ByMA, terminados en .BA).
    Si usar_adr=True, traduce a su ADR en USD antes de calcular y vuelve a
    indexar el resultado con el ticker original — evita mezclar ARS y USD
    en el mismo ranking.
    """
    if usar_adr:
        tickers_calculo = [MAPA_ADR[t] for t in tickers if t in MAPA_ADR]
        mapa_inverso = {v: k for k, v in MAPA_ADR.items()}
    else:
        tickers_calculo = tickers
        mapa_inverso = {}

    if not tickers_calculo:
        return pd.DataFrame()

    riesgo_retorno = calcular_riesgo_retorno(precios, tickers_calculo, meses)
    retorno_acum = calcular_retorno_acumulado(precios, tickers_calculo + ["SPY"], meses)
    retorno_spy = retorno_acum.get("SPY", np.nan)

    df = riesgo_retorno.copy()
    df["retorno_acumulado_pct"] = retorno_acum.reindex(df.index)
    df["alpha_vs_spy_pct"] = df["retorno_acumulado_pct"] - retorno_spy
    df["sharpe_simple"] = df["retorno_pct"] / df["volatilidad_pct"]

    # Nota: alpha_vs_spy_pct y retorno_acumulado_pct difieren en una constante
    # (el retorno de SPY), así que rankean idéntico. Se muestran ambos valores
    # porque son informativos para el usuario, pero el score solo necesita uno.
    df["score_sharpe"] = df["sharpe_simple"].rank(pct=True) * 100
    df["score_retorno"] = df["retorno_acumulado_pct"].rank(pct=True) * 100
    df["score_final"] = ((df["score_sharpe"] + df["score_retorno"]) / 2).round(1)

    df = df.dropna(subset=["score_final"]).sort_values("score_final", ascending=False)

    if mapa_inverso:
        df.index = [mapa_inverso.get(t, t) for t in df.index]

    return df


# ─────────────────────────────────────────────────────────────────
# CLASE DE CONVENIENCIA — punto de entrada único para la app
# ─────────────────────────────────────────────────────────────────

class MarketAnalyzer:
    """
    Punto de entrada único para la pestaña 'Análisis de Mercado'.

    A propósito NO hace caching interno: la app decide la política de
    refresco (st.cache_resource / st.cache_data con TTL) para no acoplar
    este módulo a Streamlit y poder reusarlo también desde Colab/notebooks.
    """

    def __init__(self, periodo: str = "3y"):
        self.periodo = periodo
        self.precios = descargar_universo(periodo)
        self.market_caps = obtener_market_caps(UNIVERSO)

    def top20(self, mercado: str, criterio: str = "market_cap") -> pd.DataFrame:
        return seleccionar_top_n(self.precios, self.market_caps, mercado, n=20, criterio=criterio)

    def ranking(self, mercado: str, meses: int = 36) -> pd.DataFrame:
        tickers = self.top20(mercado).index.tolist()
        usar_adr = (mercado == "byma")
        return construir_ranking(tickers, self.precios, meses=meses, usar_adr=usar_adr)

    def serie_acumulada(self, mercado: str, meses: int = 36) -> pd.DataFrame:
        tickers = self.top20(mercado).index.tolist()
        if mercado == "byma":
            tickers = [MAPA_ADR[t] for t in tickers if t in MAPA_ADR]
        return serie_base100(self.precios, tickers + ["SPY"], meses=meses)
