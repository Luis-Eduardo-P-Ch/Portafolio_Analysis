"""
Dividend Scanner Model
======================
Analiza dividendos pagados por acciones del NYSE/NASDAQ y ByMA.
Genera rankings de mayores y menores pagadores de dividendos.

Autor: Curso Python
Versión: 1.0
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from market_scanner_model import NYSE_TICKERS, BYMA_TICKERS

# Subconjunto de tickers más conocidos para el scanner de dividendos.
# Usar la lista entera de 200+ tickers sería muy lento con calls individuales.
NYSE_DIV_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADP", "AIG", "AMZN", "AXP",
    "BA", "BAC", "BBD", "BK", "BKNG", "BRKB", "C", "CAT",
    "CL", "COST", "CRM", "CSCO", "CVS", "CVX", "D", "DD",
    "DE", "DEO", "DHR", "DOW", "ECL", "EFX", "EQNR", "F",
    "FDX", "GE", "GILD", "GLW", "GM", "GOOGL", "GS", "HAL",
    "HD", "HON", "HPQ", "HSBC", "IBM", "INTC", "IP", "ISRG",
    "ITUB", "JNJ", "JPM", "KMB", "KO", "LLY", "LMT", "MA",
    "MCD", "MDLZ", "MDT", "META", "MMM", "MO", "MRK", "MSFT",
    "MU", "NEE", "NEM", "NFLX", "NKE", "NUE", "NVDA", "NVS",
    "ORCL", "OXY", "PBR", "PCAR", "PEP", "PFE", "PG", "PM",
    "PSX", "PYPL", "QCOM", "RIO", "RTX", "SAP", "SBUX", "SCCO",
    "SHEL", "SLB", "SPGI", "SPOT", "T", "TD", "TGT", "TJX",
    "TM", "TMUS", "TSM", "TTE", "UNH", "UNP", "USB", "V",
    "VALE", "VZ", "WFC", "WMT", "XOM",
]

BYMA_DIV_TICKERS = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "CEPU.BA",
    "ALUA.BA", "TXAR.BA", "COME.BA", "TGSU2.BA", "EDN.BA",
    "BBAR.BA", "CRES.BA", "MIRG.BA", "SUPV.BA", "LOMA.BA",
    "CVH.BA", "IRSA.BA", "METR.BA", "TECO2.BA", "VALO.BA",
    "BYMA.BA", "AGRO.BA", "RICH.BA", "SEMI.BA", "MOLI.BA",
    "ROSE.BA", "CARC.BA", "INVJ.BA", "GARO.BA", "HARG.BA",
]

TOP_N = 10


def _obtener_dividendos_bulk(tickers: list, dias: int = 365) -> pd.DataFrame:
    """
    Descarga dividendos para múltiples tickers usando yfinance.
    Retorna DataFrame con columnas: Ticker, Dividendo_Total, Dividend_Yield, Nombre.
    """
    end   = datetime.today()
    start = end - timedelta(days=dias)

    resultados = []

    # Procesamos en lotes para no saturar la API
    lote = 20
    for i in range(0, len(tickers), lote):
        sub = tickers[i : i + lote]
        for ticker_sym in sub:
            try:
                t = yf.Ticker(ticker_sym)

                # Dividendos en el período
                hist_div = t.dividends
                if hist_div is not None and not hist_div.empty:
                    # Filtrar por período
                    hist_div.index = hist_div.index.tz_localize(None) if hist_div.index.tz else hist_div.index
                    mask = (hist_div.index >= start) & (hist_div.index <= end)
                    div_periodo = hist_div[mask].sum()
                else:
                    div_periodo = 0.0

                # Dividend Yield desde info (puede ser None)
                info = t.fast_info
                try:
                    yield_val = getattr(info, "last_price", None)
                    # Usamos info completo para yield
                    full_info = t.info
                    dy = full_info.get("dividendYield", 0) or 0
                    nombre = full_info.get("shortName", ticker_sym)
                except Exception:
                    dy = 0
                    nombre = ticker_sym

                resultados.append({
                    "Ticker":          ticker_sym,
                    "Nombre":          nombre,
                    "Dividendo_Total": round(div_periodo, 4),
                    "Dividend_Yield":  round(dy * 100, 2),
                })

            except Exception:
                pass

    df = pd.DataFrame(resultados)
    if df.empty:
        return df

    # Filtrar tickers que efectivamente pagaron algún dividendo (o tienen yield > 0)
    df = df[(df["Dividendo_Total"] > 0) | (df["Dividend_Yield"] > 0)].copy()
    return df.reset_index(drop=True)


def top_dividend_payers(tickers: list, n: int = TOP_N, dias: int = 365) -> pd.DataFrame:
    """Retorna el top N de mayores pagadores de dividendos (por yield)."""
    df = _obtener_dividendos_bulk(tickers, dias)
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Nombre", "Dividendo_Total", "Dividend_Yield (%)"])

    df_sorted = df.sort_values("Dividend_Yield", ascending=False).head(n)
    df_sorted = df_sorted.rename(columns={"Dividend_Yield": "Dividend_Yield (%)"})
    return df_sorted.reset_index(drop=True)


def bottom_dividend_payers(tickers: list, n: int = TOP_N, dias: int = 365) -> pd.DataFrame:
    """Retorna el top N de menores pagadores de dividendos (los que pagan, pero menos)."""
    df = _obtener_dividendos_bulk(tickers, dias)
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Nombre", "Dividendo_Total", "Dividend_Yield (%)"])

    # Solo incluir los que efectivamente pagaron algo
    df_payers = df[df["Dividend_Yield"] > 0].copy()
    df_sorted  = df_payers.sort_values("Dividend_Yield", ascending=True).head(n)
    df_sorted  = df_sorted.rename(columns={"Dividend_Yield": "Dividend_Yield (%)"})
    return df_sorted.reset_index(drop=True)


class DividendScanner:
    """
    Escanea dividendos del mercado NYSE o ByMA y genera rankings.
    """

    def __init__(self, mercado: str = "nyse"):
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_DIV_TICKERS if mercado == "byma" else NYSE_DIV_TICKERS

    def escanear(self, dias: int = 365, progress_callback=None) -> dict:
        """
        Escanea dividendos y retorna dict con top payers y bottom payers.

        Returns
        -------
        dict {
            'top':    pd.DataFrame,   # mayores pagadores
            'bottom': pd.DataFrame,   # menores pagadores (pero que sí pagan)
            'todos':  pd.DataFrame,   # todos los que pagan dividendo
        }
        """
        if progress_callback:
            progress_callback(0.1)

        df = _obtener_dividendos_bulk(self.tickers, dias)

        if progress_callback:
            progress_callback(0.9)

        if df.empty:
            empty = pd.DataFrame(columns=["Ticker", "Nombre", "Dividendo_Total", "Dividend_Yield (%)"])
            return {"top": empty, "bottom": empty, "todos": empty}

        df_renamed = df.rename(columns={"Dividend_Yield": "Dividend_Yield (%)"})

        top    = df_renamed.sort_values("Dividend_Yield (%)", ascending=False).head(TOP_N).reset_index(drop=True)
        bottom = df_renamed[df_renamed["Dividend_Yield (%)"] > 0].sort_values("Dividend_Yield (%)").head(TOP_N).reset_index(drop=True)
        todos  = df_renamed.sort_values("Dividend_Yield (%)", ascending=False).reset_index(drop=True)

        if progress_callback:
            progress_callback(1.0)

        return {"top": top, "bottom": bottom, "todos": todos}
