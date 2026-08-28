"""
Fundamentals Scanner Model
===========================
Analiza ratios fundamentales (PER, ROE, Margen Neto) de acciones NYSE/NASDAQ y ByMA.
Genera rankings de mayores y menores valores para distintos períodos.

Autor: Curso Python
Versión: 1.0

Nota sobre los "períodos":
  Yahoo Finance expone financials trimestrales. Para simular períodos de 3, 6, 9 y 12 meses
  se usan los últimos N trimestres disponibles para calcular las métricas TTM (trailing).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Importamos listas de tickers del scanner de mercados
from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# ─────────────────────────────────────────────────────────────────
# SUBCONJUNTOS: usamos un subset razonable para no tardar demasiado
# ─────────────────────────────────────────────────────────────────

NYSE_FUND_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADP", "AIG", "AMD", "AMZN",
    "ASML", "AVGO", "AXP", "BA", "BAC", "BKNG", "BRKB", "C",
    "CAT", "CL", "COIN", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DOW", "EA", "ECL", "EFX", "F", "FDX",
    "GE", "GILD", "GLW", "GM", "GOOGL", "GS", "HAL", "HD",
    "HON", "HPQ", "HSBC", "IBM", "INTC", "ISRG", "ITUB", "JNJ",
    "JPM", "KMB", "KO", "LLY", "LMT", "MA", "MCD", "MDLZ",
    "MDT", "MELI", "META", "MMM", "MO", "MRK", "MRNA", "MSFT",
    "MU", "NEM", "NFLX", "NKE", "NOW", "NVDA", "NVS", "ORCL",
    "OXY", "PANW", "PBR", "PCAR", "PEP", "PFE", "PG", "PLTR",
    "PM", "PYPL", "QCOM", "RIO", "RTX", "SAP", "SBUX", "SCCO",
    "SHEL", "SLB", "SPGI", "SPOT", "T", "TD", "TGT", "TJX",
    "TM", "TMUS", "TSLA", "TSM", "TTE", "UNH", "UNP", "V",
    "VALE", "VZ", "WFC", "WMT", "XOM",
]

BYMA_FUND_TICKERS = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "CEPU.BA",
    "ALUA.BA", "TXAR.BA", "COME.BA", "TGSU2.BA", "EDN.BA",
    "BBAR.BA", "CRES.BA", "MIRG.BA", "SUPV.BA", "LOMA.BA",
    "CVH.BA", "IRSA.BA", "METR.BA", "TECO2.BA", "VALO.BA",
    "BYMA.BA", "AGRO.BA", "RICH.BA", "SEMI.BA", "MOLI.BA",
    "ROSE.BA", "CARC.BA", "INVJ.BA", "GARO.BA", "HARG.BA",
]

# Períodos: nombre → cantidad de trimestres a usar
PERIODOS_FUND = {
    "3 meses":  1,   # último trimestre
    "6 meses":  2,   # últimos 2 trimestres
    "9 meses":  3,   # últimos 3 trimestres
    "12 meses": 4,   # últimos 4 trimestres (TTM)
}

TOP_N = 10


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL DE DESCARGA
# ─────────────────────────────────────────────────────────────────

def _obtener_fundamentals_bulk(tickers: list, progress_callback=None) -> pd.DataFrame:
    """
    Descarga datos fundamentales para una lista de tickers.
    Devuelve DataFrame con métricas por trimestre disponible.

    Columnas resultantes:
        Ticker, Nombre, Trimestre (fecha),
        PER, ROE_pct, Margen_Neto_pct,
        Ingresos, Utilidad_Neta, Patrimonio
    """
    resultados = []
    total = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            t = yf.Ticker(sym)

            # — Precio e info básica —
            info = t.info or {}
            nombre = info.get("shortName", sym)

            # PER trailing desde info (más confiable para el valor actual)
            per_trailing = info.get("trailingPE", None)
            per_forward  = info.get("forwardPE",  None)

            # ROE trailing desde info
            roe_info = info.get("returnOnEquity", None)   # decimal (ej: 0.15 = 15%)

            # Margen neto trailing desde info
            margen_info = info.get("profitMargins", None)  # decimal

            # — Income Statement trimestral —
            try:
                inc = t.quarterly_income_stmt
            except Exception:
                inc = None

            # — Balance Sheet trimestral —
            try:
                bal = t.quarterly_balance_sheet
            except Exception:
                bal = None

            # Si tenemos estados financieros, construimos serie por trimestre
            if inc is not None and not inc.empty:
                cols = inc.columns[:4]  # máx 4 trimestres recientes

                for j, col in enumerate(cols):
                    trimestre = col if hasattr(col, 'date') else col

                    # Ingresos
                    ingresos = _get_row(inc, col, [
                        "Total Revenue", "Revenue", "Net Revenue",
                        "TotalRevenue",
                    ])

                    # Utilidad neta
                    util_neta = _get_row(inc, col, [
                        "Net Income", "NetIncome",
                        "Net Income Common Stockholders",
                    ])

                    # Patrimonio neto del balance (para ROE)
                    patrimonio = None
                    if bal is not None and not bal.empty:
                        # Buscamos la columna de balance más cercana al trimestre
                        if col in bal.columns:
                            bal_col = col
                        else:
                            # Tomar la columna más cercana
                            bal_col = bal.columns[min(j, len(bal.columns)-1)]

                        patrimonio = _get_row(bal, bal_col, [
                            "Stockholders Equity",
                            "Total Stockholder Equity",
                            "Total Equity Gross Minority Interest",
                            "Common Stock Equity",
                        ])

                    # — Cálculo de ratios para este trimestre —
                    # PER: usamos el de info para todos (es el más reciente y fiable)
                    # Para trimestres pasados aproximamos con EPS si disponible
                    eps_trim = _get_row(inc, col, ["Basic EPS", "Diluted EPS", "EPS"])
                    precio_actual = info.get("currentPrice") or info.get("regularMarketPrice")

                    per_calc = None
                    if j == 0:
                        # Trimestre más reciente: usar PER de info
                        per_calc = per_trailing
                    elif eps_trim and precio_actual:
                        # Aproximación: precio / (EPS trimestral * 4)
                        eps_anual_aprox = eps_trim * 4
                        if eps_anual_aprox and eps_anual_aprox > 0:
                            per_calc = round(precio_actual / eps_anual_aprox, 2)

                    # ROE = Utilidad_Neta (anualizada) / Patrimonio
                    roe_calc = None
                    if j == 0:
                        roe_calc = roe_info * 100 if roe_info is not None else None
                    elif util_neta is not None and patrimonio and patrimonio > 0:
                        roe_anual = util_neta * 4
                        roe_calc  = round((roe_anual / patrimonio) * 100, 2)

                    # Margen neto = Utilidad_Neta / Ingresos
                    margen_calc = None
                    if j == 0:
                        margen_calc = margen_info * 100 if margen_info is not None else None
                    elif util_neta is not None and ingresos and ingresos > 0:
                        margen_calc = round((util_neta / ingresos) * 100, 2)

                    resultados.append({
                        "Ticker":          sym,
                        "Nombre":          nombre,
                        "Trimestre_idx":   j,          # 0 = más reciente
                        "Trimestre_fecha": str(trimestre)[:10],
                        "PER":             _safe_round(per_calc),
                        "ROE_pct":         _safe_round(roe_calc),
                        "Margen_Neto_pct": _safe_round(margen_calc),
                    })

            else:
                # Sin estados financieros: solo usamos los valores de info
                resultados.append({
                    "Ticker":          sym,
                    "Nombre":          nombre,
                    "Trimestre_idx":   0,
                    "Trimestre_fecha": "N/A",
                    "PER":             _safe_round(per_trailing),
                    "ROE_pct":         _safe_round(roe_info * 100 if roe_info else None),
                    "Margen_Neto_pct": _safe_round(margen_info * 100 if margen_info else None),
                })

        except Exception:
            pass

        if progress_callback:
            progress_callback((i + 1) / total)

    if not resultados:
        return pd.DataFrame()

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _get_row(df: pd.DataFrame, col, row_names: list):
    """Busca el primer row_name disponible en el DataFrame y devuelve el valor de col."""
    for rn in row_names:
        if rn in df.index:
            try:
                val = df.loc[rn, col]
                if pd.notna(val):
                    return float(val)
            except Exception:
                pass
    return None


def _safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN DE AGREGACIÓN POR PERÍODO
# ─────────────────────────────────────────────────────────────────

def _agregar_por_periodo(df_raw: pd.DataFrame, n_trimestres: int, metrica: str) -> pd.Series:
    """
    Para cada ticker, promedia la métrica usando los últimos n_trimestres disponibles.
    Devuelve una Serie {ticker: valor_promedio}.
    """
    if df_raw.empty:
        return pd.Series(dtype=float)

    # Filtrar solo los trimestres que caen dentro del período
    df_filt = df_raw[df_raw["Trimestre_idx"] < n_trimestres].copy()

    # Agrupar por ticker y promediar la métrica (ignorando NaN)
    serie = (
        df_filt.groupby("Ticker")[metrica]
        .mean()
        .dropna()
    )
    return serie


def _nombres_dict(df_raw: pd.DataFrame) -> dict:
    """Devuelve dict {ticker: nombre} del DataFrame raw."""
    return df_raw.drop_duplicates("Ticker").set_index("Ticker")["Nombre"].to_dict()


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN DE RANKING
# ─────────────────────────────────────────────────────────────────

def _build_ranking(serie: pd.Series, nombres: dict, n: int, ascending: bool, metrica_col: str) -> pd.DataFrame:
    """Construye DataFrame de ranking a partir de una Serie."""
    if serie.empty:
        return pd.DataFrame(columns=["Ticker", "Nombre", metrica_col])

    ranked = serie.sort_values(ascending=ascending).head(n)
    df = ranked.reset_index()
    df.columns = ["Ticker", metrica_col]
    df["Nombre"] = df["Ticker"].map(nombres).fillna(df["Ticker"])
    df[metrica_col] = df[metrica_col].round(2)
    return df[["Ticker", "Nombre", metrica_col]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class FundamentalsScanner:
    """
    Escanea ratios fundamentales (PER, ROE, Margen Neto) del mercado NYSE o ByMA
    y genera rankings top/bottom para distintos períodos.
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_FUND_TICKERS if mercado == "byma" else NYSE_FUND_TICKERS
        self._df_raw = None

    def _asegurar_datos(self, progress_callback=None):
        """Descarga datos si no están cacheados."""
        if self._df_raw is None or self._df_raw.empty:
            self._df_raw = _obtener_fundamentals_bulk(self.tickers, progress_callback)

    # ── API pública ──────────────────────────────────────────────

    def escanear_per(self, progress_callback=None) -> dict:
        """
        Escanea PER para todos los períodos.

        Returns
        -------
        dict {
            '3 meses':  {'top': df, 'bottom': df},
            '6 meses':  {'top': df, 'bottom': df},
            ...
        }
        """
        self._asegurar_datos(progress_callback)
        return self._escanear_metrica("PER", progress_callback=None)

    def escanear_roe(self, progress_callback=None) -> dict:
        """Escanea ROE (%) para todos los períodos."""
        self._asegurar_datos(progress_callback)
        return self._escanear_metrica("ROE_pct", progress_callback=None)

    def escanear_margen(self, progress_callback=None) -> dict:
        """Escanea Margen Neto (%) para todos los períodos."""
        self._asegurar_datos(progress_callback)
        return self._escanear_metrica("Margen_Neto_pct", progress_callback=None)

    def escanear_todo(self, progress_callback=None) -> dict:
        """
        Descarga datos UNA sola vez y calcula PER, ROE y Margen para todos los períodos.

        Returns
        -------
        dict {
            'per':    { '3 meses': {'top': df, 'bottom': df}, ... },
            'roe':    { ... },
            'margen': { ... },
        }
        """
        self._asegurar_datos(progress_callback)
        return {
            "per":    self._escanear_metrica("PER"),
            "roe":    self._escanear_metrica("ROE_pct"),
            "margen": self._escanear_metrica("Margen_Neto_pct"),
        }

    # ── Internos ─────────────────────────────────────────────────

    def _escanear_metrica(self, metrica: str, progress_callback=None) -> dict:
        """Calcula ranking top/bottom para cada período definido en PERIODOS_FUND."""
        nombres = _nombres_dict(self._df_raw)
        col_label = {
            "PER":             "PER",
            "ROE_pct":         "ROE (%)",
            "Margen_Neto_pct": "Margen Neto (%)",
        }[metrica]

        resultado = {}
        for periodo_nombre, n_trim in PERIODOS_FUND.items():
            serie = _agregar_por_periodo(self._df_raw, n_trim, metrica)

            # Filtrar valores extremos / negativos para PER
            if metrica == "PER":
                serie = serie[serie > 0]   # PER negativo o 0 no es significativo

            top    = _build_ranking(serie, nombres, TOP_N, ascending=False, metrica_col=col_label)
            bottom = _build_ranking(serie, nombres, TOP_N, ascending=True,  metrica_col=col_label)

            resultado[periodo_nombre] = {"top": top, "bottom": bottom}

        return resultado
