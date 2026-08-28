"""
Contrarian Multifactor Model
=============================
Identifica acciones que más cayeron en precio en un período dado
y las ranquea usando un modelo multifactor (Momentum + Valoración + Calidad + Dividendos).

Lógica "contrarian":
  1. Se toman las N acciones con mayor caída de precio en el período seleccionado.
  2. A ese subconjunto se le aplica el mismo scoring multifactor del Smart Ranking.
  3. El resultado muestra las "caídas con mejores fundamentos": candidatas a rebote.

Factores y pesos:
  - Momentum  (6m + 12m promedio) : 40%
  - Valoración (PER inverso)       : 20%
  - Calidad   (ROA + Margen + ROE) : 30%
  - Dividendos (Dividend Yield)    : 10%

Autor: Curso Python
Versión: 1.1  — compatible con yfinance >= 1.0 (manejo de MultiIndex)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

PESOS_C = {
    "momentum":   0.40,
    "valoracion": 0.20,
    "calidad":    0.30,
    "dividendos": 0.10,
}

# Cuántas acciones "más caídas" tomar como universo contrarian
UNIVERSO_CAIDAS = 40

# Top N final a mostrar
TOP_N = 20

PERIODOS_CONTRARIAN = {
    "1 mes":    30,
    "3 meses":  90,
    "6 meses":  180,
    "12 meses": 365,
}


# ─────────────────────────────────────────────────────────────────
# HELPER: normalizar DataFrame de yfinance (maneja MultiIndex)
# ─────────────────────────────────────────────────────────────────

def _normalizar_close(data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    yfinance >= 1.0 devuelve MultiIndex en columnas: ("Close", "AAPL").
    Esta función lo normaliza a un DataFrame simple con tickers como columnas.
    También maneja el caso de un solo ticker (columna "Close" simple).
    """
    if data is None or data.empty:
        return pd.DataFrame()

    cols = data.columns

    # Caso MultiIndex: ("Close", "AAPL"), ("Open", "AAPL"), ...
    if isinstance(cols, pd.MultiIndex):
        # Nivel 0 tiene el tipo de dato, nivel 1 tiene el ticker
        if "Close" in cols.get_level_values(0):
            df = data["Close"].copy()
        elif "Close" in cols.get_level_values(1):
            df = data.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    # Caso un solo ticker o columnas planas
    elif "Close" in cols:
        if len(tickers) == 1:
            df = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            df = data[["Close"]].copy()
    else:
        return pd.DataFrame()

    # Asegurarse de que las columnas sean strings (no tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Eliminar columnas totalmente vacías
    df = df.dropna(axis=1, how="all")
    return df


# ─────────────────────────────────────────────────────────────────
# DESCARGA DE PRECIOS
# ─────────────────────────────────────────────────────────────────

def _descargar_precios_ct(tickers: list, dias: int) -> pd.DataFrame:
    """Descarga precios de cierre para la lista de tickers en los últimos N días."""
    end   = datetime.today()
    start = end - timedelta(days=dias + 30)

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        return _normalizar_close(data, tickers)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# CALCULAR VARIACIÓN DE PRECIO EN EL PERÍODO
# ─────────────────────────────────────────────────────────────────

def _calcular_variacion_ct(precios: pd.DataFrame, dias: int) -> pd.Series:
    """
    Variación porcentual entre el precio de hace `dias` días
    y el precio más reciente disponible para cada ticker.
    """
    if precios.empty:
        return pd.Series(dtype=float)

    precio_actual = precios.iloc[-1]

    fecha_objetivo = precios.index[-1] - timedelta(days=dias)
    historico = precios[precios.index <= fecha_objetivo]

    if historico.empty:
        return pd.Series(dtype=float)

    precio_pasado = historico.iloc[-1]

    variacion = ((precio_actual - precio_pasado) / precio_pasado) * 100
    return variacion.dropna()


# ─────────────────────────────────────────────────────────────────
# MOMENTUM 6m / 12m
# ─────────────────────────────────────────────────────────────────

def _calcular_momentum(tickers: list) -> pd.DataFrame:
    """Calcula momentum 6m y 12m para una lista de tickers."""
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
        precios = _normalizar_close(data, tickers)
        if precios.empty:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    resultados = []
    precio_actual = precios.iloc[-1]

    for sym in precios.columns:
        serie = precios[sym].dropna()
        if len(serie) < 60:
            continue

        fecha_6m  = serie.index[-1] - timedelta(days=180)
        hist_6m   = serie[serie.index <= fecha_6m]
        fecha_12m = serie.index[-1] - timedelta(days=365)
        hist_12m  = serie[serie.index <= fecha_12m]

        p_actual = float(precio_actual[sym])
        mom_6m   = None
        mom_12m  = None

        if not hist_6m.empty:
            mom_6m  = (p_actual - float(hist_6m.iloc[-1])) / float(hist_6m.iloc[-1]) * 100
        if not hist_12m.empty:
            mom_12m = (p_actual - float(hist_12m.iloc[-1])) / float(hist_12m.iloc[-1]) * 100

        if mom_6m is not None and mom_12m is not None:
            resultados.append({
                "Ticker": str(sym),
                "mom_6m":  round(mom_6m,  2),
                "mom_12m": round(mom_12m, 2),
                "mom_avg": round((mom_6m + mom_12m) / 2, 2),
            })

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────

def _calcular_fundamentals(tickers: list, progress_callback=None) -> pd.DataFrame:
    """Descarga PER, ROA, Margen, ROE y Dividend Yield."""
    resultados = []
    total = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            t    = yf.Ticker(sym)
            info = t.info or {}

            nombre = info.get("shortName", sym)

            per = info.get("trailingPE", None)
            if per is not None:
                per = float(per)
                if per <= 0 or per > 1000:
                    per = None

            roa    = info.get("returnOnAssets",  None)
            roe    = info.get("returnOnEquity",  None)
            margen = info.get("profitMargins",   None)

            roa_pct    = round(roa    * 100, 2) if roa    is not None else None
            roe_pct    = round(roe    * 100, 2) if roe    is not None else None
            margen_pct = round(margen * 100, 2) if margen is not None else None

            dy     = info.get("dividendYield", None)
            dy_pct = round(dy * 100, 2) if dy is not None else 0.0

            resultados.append({
                "Ticker":     str(sym),
                "Nombre":     nombre,
                "PER":        per,
                "ROA_pct":    roa_pct,
                "ROE_pct":    roe_pct,
                "Margen_pct": margen_pct,
                "DY_pct":     dy_pct,
            })

        except Exception:
            pass

        if progress_callback:
            progress_callback((i + 1) / total)

    return pd.DataFrame(resultados)


# ─────────────────────────────────────────────────────────────────
# PERCENTILE RANK
# ─────────────────────────────────────────────────────────────────

def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(method="average", ascending=ascending, na_option="keep")
    n      = series.notna().sum()
    pct    = (ranked - 1) / (n - 1) * 100 if n > 1 else ranked * 0
    return pct.round(2)


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class ContrarianScanner:
    """
    Modelo Contrarian Multifactor:
      1. Filtra las acciones con mayor caída de precio en el período indicado.
      2. Aplica scoring multifactor a ese universo.
      3. Retorna un ranking de "caídas con mejores fundamentos".
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_TICKERS if mercado == "byma" else NYSE_TICKERS

    def calcular(self, dias: int = 180, progress_callback=None) -> pd.DataFrame:
        """
        Ejecuta el modelo completo.

        Parameters
        ----------
        dias : int — período de caída a analizar (30, 90, 180, 365)
        progress_callback : callable(float, str) opcional

        Returns
        -------
        pd.DataFrame ordenado por smart_score descendente
        """

        # ── 1. Descargar precios y calcular caída del período ────
        if progress_callback:
            progress_callback(0.05, "Descargando precios del mercado...")

        precios = _descargar_precios_ct(self.tickers, dias)

        if precios.empty:
            return pd.DataFrame()

        variacion = _calcular_variacion_ct(precios, dias)

        if variacion.empty:
            return pd.DataFrame()

        # ── 2. Tomar las N acciones con mayor caída ──────────────
        caidas_negativas = variacion[variacion < 0]
        if caidas_negativas.empty:
            caidas_negativas = variacion  # fallback: tomar todas

        caidas = caidas_negativas.sort_values().head(UNIVERSO_CAIDAS)
        tickers_caidos = [str(t) for t in caidas.index.tolist()]

        if not tickers_caidos:
            return pd.DataFrame()

        if progress_callback:
            progress_callback(0.20, f"Analizando las {len(tickers_caidos)} acciones más caídas...")

        df_caidas = pd.DataFrame({
            "Ticker":        tickers_caidos,
            "caida_periodo": [round(float(caidas.iloc[i]), 2) for i in range(len(caidas))],
        })

        # ── 3. Momentum ──────────────────────────────────────────
        if progress_callback:
            progress_callback(0.30, "Calculando momentum (6m / 12m)...")

        df_mom = _calcular_momentum(tickers_caidos)

        # ── 4. Fundamentals ──────────────────────────────────────
        def _prog_fund(val):
            if progress_callback:
                pct = 0.35 + val * 0.55
                progress_callback(pct, f"Descargando fundamentals... {int(val*100)}%")

        df_fund = _calcular_fundamentals(tickers_caidos, progress_callback=_prog_fund)

        if progress_callback:
            progress_callback(0.92, "Calculando scores multifactor...")

        # ── 5. Merge ─────────────────────────────────────────────
        if df_fund.empty:
            return pd.DataFrame()

        # Si no hay momentum, hacer merge solo con fund y usar caída como proxy
        if df_mom.empty:
            df = pd.merge(df_caidas, df_fund, on="Ticker", how="inner")
            df["mom_6m"]  = np.nan
            df["mom_12m"] = np.nan
            df["mom_avg"] = df["caida_periodo"]  # usar caída del período como proxy
        else:
            df = pd.merge(df_caidas, df_mom,  on="Ticker", how="left")
            df = pd.merge(df,        df_fund, on="Ticker", how="inner")
            # Para tickers sin momentum calculado, usar caída como proxy
            mask = df["mom_avg"].isna()
            df.loc[mask, "mom_avg"] = df.loc[mask, "caida_periodo"]

        if df.empty:
            return pd.DataFrame()

        # ── 6. Scores por factor ──────────────────────────────────
        df["score_momentum"]   = _percentile_rank(df["mom_avg"], ascending=True)
        df["score_valoracion"] = _percentile_rank(df["PER"],     ascending=False).fillna(50)

        scores_calidad = []
        for col in ["ROA_pct", "ROE_pct", "Margen_pct"]:
            s = _percentile_rank(df[col], ascending=True).fillna(50)
            scores_calidad.append(s)
        df["score_calidad"] = pd.concat(scores_calidad, axis=1).mean(axis=1).round(2)

        df["score_dividendos"] = _percentile_rank(df["DY_pct"], ascending=True).fillna(0)

        # ── 7. Smart Score ponderado ─────────────────────────────
        df["smart_score"] = (
            df["score_momentum"]   * PESOS_C["momentum"]   +
            df["score_valoracion"] * PESOS_C["valoracion"] +
            df["score_calidad"]    * PESOS_C["calidad"]    +
            df["score_dividendos"] * PESOS_C["dividendos"]
        ).round(1)

        # ── 8. Ordenar y devolver Top N ──────────────────────────
        df = df.sort_values("smart_score", ascending=False).head(TOP_N).reset_index(drop=True)
        df.index = df.index + 1

        if progress_callback:
            progress_callback(1.0, "¡Listo!")

        cols = [
            "Ticker", "Nombre", "caida_periodo",
            "smart_score",
            "score_momentum", "score_valoracion", "score_calidad", "score_dividendos",
            "mom_6m", "mom_12m",
            "PER", "ROA_pct", "ROE_pct", "Margen_pct", "DY_pct",
        ]
        return df[[c for c in cols if c in df.columns]]
