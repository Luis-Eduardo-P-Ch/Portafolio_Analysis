"""
Multifactor Ranking Model
=========================
Modelo de ranking cuantitativo multifactor inspirado en factor investing.

Factores y pesos:
  - Momentum  (6m + 12m promedio) : 40%
  - Valoración (PER inverso)       : 20%
  - Calidad   (ROA + Margen + ROE) : 30%
  - Dividendos (Dividend Yield)    : 10%

Metodología:
  1. Para cada factor se calcula un percentil rank (0–100) sobre el universo.
  2. Se aplican los pesos para obtener el Smart Score final (0–100).
  3. Se presentan los Top 20 con breakdown por factor.

Autor: Curso Python
Versión: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# ─────────────────────────────────────────────────────────────────
# UNIVERSOS
# ─────────────────────────────────────────────────────────────────

NYSE_MF_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADP", "AMD", "AMZN", "ASML",
    "AVGO", "AXP", "BA", "BAC", "BKNG", "C", "CAT", "CL",
    "COIN", "COST", "CRM", "CSCO", "CVS", "CVX", "DE", "DHR",
    "DOW", "EA", "ECL", "F", "FDX", "GE", "GILD", "GLW",
    "GM", "GOOGL", "GS", "HAL", "HD", "HON", "HPQ", "IBM",
    "INTC", "ISRG", "ITUB", "JNJ", "JPM", "KMB", "KO", "LLY",
    "LMT", "MA", "MCD", "MDLZ", "MDT", "MELI", "META", "MMM",
    "MO", "MRK", "MRNA", "MSFT", "MU", "NEM", "NFLX", "NKE",
    "NOW", "NVDA", "NVS", "ORCL", "OXY", "PANW", "PBR", "PEP",
    "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RIO", "RTX",
    "SAP", "SBUX", "SCCO", "SHEL", "SLB", "SPGI", "SPOT", "T",
    "TD", "TGT", "TJX", "TM", "TMUS", "TSLA", "TSM", "TTE",
    "UNH", "UNP", "V", "VALE", "VZ", "WFC", "WMT", "XOM",
]

BYMA_MF_TICKERS = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "CEPU.BA",
    "ALUA.BA", "TXAR.BA", "COME.BA", "TGSU2.BA", "EDN.BA",
    "BBAR.BA", "CRES.BA", "MIRG.BA", "SUPV.BA", "LOMA.BA",
    "CVH.BA", "IRSA.BA", "METR.BA", "TECO2.BA", "VALO.BA",
    "BYMA.BA", "AGRO.BA", "RICH.BA", "SEMI.BA", "MOLI.BA",
    "ROSE.BA", "CARC.BA", "INVJ.BA", "GARO.BA", "HARG.BA",
]

# ─────────────────────────────────────────────────────────────────
# PESOS DEL MODELO
# ─────────────────────────────────────────────────────────────────

PESOS = {
    "momentum":   0.40,
    "valoracion": 0.20,
    "calidad":    0.30,
    "dividendos": 0.10,
}

TOP_N = 20


# ─────────────────────────────────────────────────────────────────
# DESCARGA DE PRECIOS (MOMENTUM)
# ─────────────────────────────────────────────────────────────────

def _descargar_momentum(tickers: list) -> pd.DataFrame:
    """
    Descarga 13 meses de precios y calcula retorno acumulado a 6m y 12m.
    Devuelve DataFrame con columnas: Ticker, mom_6m, mom_12m, mom_score
    """
    end   = datetime.today()
    start = end - timedelta(days=400)

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            return pd.DataFrame()

        precios = data["Close"] if len(tickers) > 1 else data[["Close"]].rename(columns={"Close": tickers[0]})
        precios = precios.dropna(axis=1, how="all")

    except Exception:
        return pd.DataFrame()

    resultados = []
    precio_actual = precios.iloc[-1]

    for sym in precios.columns:
        serie = precios[sym].dropna()
        if len(serie) < 60:
            continue

        # Precio hace 6 meses
        fecha_6m  = serie.index[-1] - timedelta(days=180)
        hist_6m   = serie[serie.index <= fecha_6m]
        # Precio hace 12 meses
        fecha_12m = serie.index[-1] - timedelta(days=365)
        hist_12m  = serie[serie.index <= fecha_12m]

        p_actual = float(precio_actual[sym])

        mom_6m  = None
        mom_12m = None

        if not hist_6m.empty:
            mom_6m  = (p_actual - float(hist_6m.iloc[-1])) / float(hist_6m.iloc[-1]) * 100
        if not hist_12m.empty:
            mom_12m = (p_actual - float(hist_12m.iloc[-1])) / float(hist_12m.iloc[-1]) * 100

        if mom_6m is not None and mom_12m is not None:
            resultados.append({
                "Ticker": sym,
                "mom_6m":  round(mom_6m,  2),
                "mom_12m": round(mom_12m, 2),
                "mom_avg": round((mom_6m + mom_12m) / 2, 2),
            })

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# DESCARGA DE FUNDAMENTALS (VALORACIÓN + CALIDAD + DIVIDENDOS)
# ─────────────────────────────────────────────────────────────────

def _descargar_fundamentals(tickers: list, progress_callback=None) -> pd.DataFrame:
    """
    Descarga PER, ROA, Margen Neto, ROE y Dividend Yield para cada ticker.
    Devuelve DataFrame con una fila por ticker.
    """
    resultados = []
    total = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            t    = yf.Ticker(sym)
            info = t.info or {}

            nombre = info.get("shortName", sym)

            # ── Valoración ───────────────────────────────────────
            per = info.get("trailingPE", None)
            # PER muy alto o negativo → lo tratamos como None
            if per is not None:
                per = float(per)
                if per <= 0 or per > 1000:
                    per = None

            # ── Calidad ──────────────────────────────────────────
            roa    = info.get("returnOnAssets",  None)   # decimal
            roe    = info.get("returnOnEquity",  None)   # decimal
            margen = info.get("profitMargins",   None)   # decimal

            roa_pct    = round(roa    * 100, 2) if roa    is not None else None
            roe_pct    = round(roe    * 100, 2) if roe    is not None else None
            margen_pct = round(margen * 100, 2) if margen is not None else None

            # ── Dividendos ───────────────────────────────────────
            dy = info.get("dividendYield", None)
            dy_pct = round(dy * 100, 2) if dy is not None else 0.0

            resultados.append({
                "Ticker":    sym,
                "Nombre":    nombre,
                "PER":       per,
                "ROA_pct":   roa_pct,
                "ROE_pct":   roe_pct,
                "Margen_pct": margen_pct,
                "DY_pct":    dy_pct,
            })

        except Exception:
            pass

        if progress_callback:
            progress_callback((i + 1) / total)

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# PERCENTILE RANK (0–100)
# ─────────────────────────────────────────────────────────────────

def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Convierte una Serie numérica a percentil rank de 0 a 100.
    ascending=True  → mayor valor = mayor percentil (ej: momentum, ROA)
    ascending=False → menor valor = mayor percentil (ej: PER bajo = mejor)
    """
    ranked = series.rank(method="average", ascending=ascending, na_option="keep")
    n      = series.notna().sum()
    pct    = (ranked - 1) / (n - 1) * 100 if n > 1 else ranked * 0
    return pct.round(2)


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class MultifactorScanner:
    """
    Construye un ranking multifactor ponderado para NYSE o ByMA.

    Smart Score = 0.40 * Mom + 0.20 * Val + 0.30 * Cal + 0.10 * Div
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_MF_TICKERS if mercado == "byma" else NYSE_MF_TICKERS

    def calcular(self, progress_callback=None) -> pd.DataFrame:
        """
        Ejecuta el modelo completo y devuelve un DataFrame con:
            Ticker, Nombre,
            mom_6m, mom_12m,               ← valores raw
            PER, ROA_pct, ROE_pct, Margen_pct, DY_pct,
            score_momentum, score_valoracion, score_calidad, score_dividendos,
            smart_score
        Ordenado por smart_score descendente (Top N primero).
        """

        # ── 1. Momentum ──────────────────────────────────────────
        if progress_callback:
            progress_callback(0.05, "Descargando precios (momentum)...")
        df_mom = _descargar_momentum(self.tickers)

        if progress_callback:
            progress_callback(0.35, "Descargando fundamentals...")

        # ── 2. Fundamentals ──────────────────────────────────────
        tickers_con_mom = df_mom["Ticker"].tolist() if not df_mom.empty else self.tickers

        def _prog_fund(val):
            if progress_callback:
                pct = 0.35 + val * 0.55
                progress_callback(pct, f"Analizando fundamentals... {int(val*100)}%")

        df_fund = _descargar_fundamentals(self.tickers, progress_callback=_prog_fund)

        if progress_callback:
            progress_callback(0.92, "Calculando scores...")

        # ── 3. Merge ─────────────────────────────────────────────
        if df_mom.empty or df_fund.empty:
            return pd.DataFrame()

        df = pd.merge(df_mom, df_fund, on="Ticker", how="inner")
        if df.empty:
            return pd.DataFrame()

        # ── 4. Scores por factor (percentil rank 0–100) ──────────

        # Momentum: mayor retorno → mejor
        df["score_momentum"] = _percentile_rank(df["mom_avg"], ascending=True)

        # Valoración: menor PER → mejor (se invierte)
        df["score_valoracion"] = _percentile_rank(df["PER"], ascending=False)
        # Si PER es NaN → score neutro 50
        df["score_valoracion"] = df["score_valoracion"].fillna(50)

        # Calidad: promedio de ROA, ROE y Margen (cada uno mayor → mejor)
        scores_calidad = []
        for col in ["ROA_pct", "ROE_pct", "Margen_pct"]:
            s = _percentile_rank(df[col], ascending=True).fillna(50)
            scores_calidad.append(s)
        df["score_calidad"] = pd.concat(scores_calidad, axis=1).mean(axis=1).round(2)

        # Dividendos: mayor yield → mejor
        df["score_dividendos"] = _percentile_rank(df["DY_pct"], ascending=True).fillna(0)

        # ── 5. Smart Score ponderado ─────────────────────────────
        df["smart_score"] = (
            df["score_momentum"]   * PESOS["momentum"]   +
            df["score_valoracion"] * PESOS["valoracion"] +
            df["score_calidad"]    * PESOS["calidad"]    +
            df["score_dividendos"] * PESOS["dividendos"]
        ).round(1)

        # ── 6. Ordenar y devolver Top N ──────────────────────────
        df = df.sort_values("smart_score", ascending=False).head(TOP_N).reset_index(drop=True)
        df.index = df.index + 1  # ranking desde 1

        if progress_callback:
            progress_callback(1.0, "¡Listo!")

        # Columnas finales limpias
        cols = [
            "Ticker", "Nombre",
            "smart_score",
            "score_momentum", "score_valoracion", "score_calidad", "score_dividendos",
            "mom_6m", "mom_12m",
            "PER", "ROA_pct", "ROE_pct", "Margen_pct", "DY_pct",
        ]
        return df[[c for c in cols if c in df.columns]]
