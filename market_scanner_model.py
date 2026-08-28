"""
Market Scanner Model
====================
Analiza las mayores caídas y mayores subidas de acciones en ByMA y NYSE/NASDAQ
para períodos de 1, 3, 6 y 12 meses.

Autor: Curso Python
Versión: 2.0
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# LISTAS DE ACTIVOS POR MERCADO
# ─────────────────────────────────────────────────────────────────

BYMA_TICKERS = [
    "A3.BA", "A3D.BA", "AGRO.BA", "AGROD.BA", "ALUA.BA", "ALUAD.BA",
    "AUSO.BA", "BBAR.BA", "BHIP.BA", "BHIPD.BA", "BMA.BA", "BOLT.BA",
    "BPAT.BA", "BYMA.BA", "CADO.BA", "CAPX.BA", "CARC.BA", "CECO2.BA",
    "CELU.BA", "CEPU.BA", "CEPUD.BA", "CGPA2.BA", "COME.BA", "CRES.BA",
    "CTIO.BA", "CVH.BA", "DGCE.BA", "DGCU2.BA", "ECOG.BA", "ECOGD.BA",
    "EDN.BA", "EDND.BA", "FERR.BA", "FIPL.BA", "GAMI.BA", "GARO.BA",
    "GBAN.BA", "GCDI.BA", "GCLA.BA", "GGAL.BA", "GGALD.BA", "HARG.BA",
    "HAVA.BA", "HSAT.BA", "IEB.BA", "INTR.BA", "INVJ.BA", "IRS2W.BA",
    "IRSA.BA", "IRSAD.BA", "LEDE.BA", "LOMA.BA", "LONG.BA", "METR.BA",
    "MIRG.BA", "MOLA.BA", "MOLI.BA", "MORI.BA", "OEST.BA", "PAMP.BA",
    "PAMPD.BA", "PATA.BA", "RAGH.BA", "RICH.BA", "RIGO.BA", "ROSE.BA",
    "SAMI.BA", "SEMI.BA", "SUPV.BA", "TECO2.BA", "TGN4D.BA", "TGSU2.BA",
    "TRAN.BA", "TXAR.BA", "VALO.BA", "YPFD.BA", "YPFDD.BA",
]

NYSE_TICKERS = [
    "AAL", "AAP", "AAPL", "ABBV", "ABEV", "ABNB", "ABT", "ACN",
    "ACWI", "ADBE", "ADP", "AEM", "AI", "AIG", "ALAB", "AMAT",
    "AMD", "AMX", "AMZN", "ARKK", "ARM", "ASML", "ASTS", "AVGO",
    "AXP", "AZN", "BA", "BABA", "BAK", "BB", "BBD", "BIDU", "BIIB",
    "BIOX", "BITF", "BK", "BKNG", "BKR", "BMNR", "BRKB", "BX",
    "C", "CAAP", "CAH", "CAT", "CDE", "CEG", "CIBR", "CL", "CLS",
    "COIN", "COPX", "COST", "CRM", "CRWV", "CSCO", "CVS", "CVX", "CX",
    "DAL", "DD", "DE", "DECK", "DEO", "DHR", "DIA", "DOCU", "DOW",
    "EA", "EBAY", "ECL", "EEM", "EFA", "EFX", "EQNR", "ESGU",
    "ETHA", "ETSY", "EWZ", "F", "FDX", "FNMA", "FSLR", "FXI",
    "GDX", "GE", "GFI", "GILD", "GLD", "GLOB", "GLW", "GM",
    "GOOGL", "GPRK", "GRMN", "GS", "GT", "HAL", "HD", "HL",
    "HMC", "HMY", "HOG", "HON", "HOOD", "HPQ", "HSBC", "HSY",
    "HUT", "HWM", "IBB", "IBIT", "IBM", "IBN", "IEMG", "IEUR",
    "IFF", "ILF", "INFY", "INTC", "IP", "IREN", "ISRG", "ITA",
    "ITUB", "IVE", "IVV", "IVW", "IWM", "JD", "JMIA", "JNJ",
    "JOYY", "JPM", "KEP", "KMB", "KO", "LAC", "LAR", "LLY",
    "LMT", "LRCX", "LVS", "MA", "MCD", "MDLZ", "MDT", "MELI",
    "META", "MFG", "MMM", "MO", "MOS", "MRK", "MRNA", "MRVL",
    "MSFT", "MSI", "MSTR", "MU", "MUFG", "MUX", "NEM", "NFLX",
    "NG", "NIO", "NKE", "NOW", "NTES", "NU", "NUE", "NVDA",
    "NVS", "NXE", "OKLO", "ORCL", "ORLY", "OXY", "PAAS", "PAGS",
    "PANW", "PATH", "PBR", "PCAR", "PDD", "PEP", "PFE", "PG",
    "PINS", "PKS", "PLTR", "PM", "PSQ", "PSX", "PYPL", "QCOM",
    "QQQ", "RACE", "RBLX", "RGTI", "RIO", "RIOT", "RKLB", "ROKU",
    "RTX", "SAN", "SAP", "SATL", "SBUX", "SCCO", "SE", "SH",
    "SHEL", "SHOP", "SID", "SIEGY", "SLB", "SLV", "SMH", "SNAP",
    "SNOW", "SONY", "SPCE", "SPGI", "SPHQ", "SPOT", "SPXL", "SPY",
    "STLA", "STNE", "SUZ", "T", "TCOM", "TD", "TEAM", "TEM",
    "TGT", "TJX", "TM", "TMUS", "TQQQ", "TRIP", "TSLA", "TSM",
    "TTE", "TWLO", "UAL", "UBER", "UL", "UNH", "UNP", "UPST",
    "URA", "URBN", "USB", "USO", "V", "VALE", "VEA", "VIG",
    "VIST", "VRSN", "VST", "VXX", "VZ", "WFC", "WMT", "XLB",
    "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
    "XLV", "XLY", "XOM", "XP", "XPEV", "ZM",
]

PERIODOS = {
    "1 mes":   30,
    "3 meses": 90,
    "6 meses": 180,
    "12 meses": 365,
}

TOP_N = 10


# ─────────────────────────────────────────────────────────────────
# FUNCIONES DE DESCARGA Y CÁLCULO
# ─────────────────────────────────────────────────────────────────

def _descargar_precios(tickers: list, dias: int) -> pd.DataFrame:
    """Descarga precios de cierre para una lista de tickers en los últimos N días."""
    end = datetime.today()
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
        if data.empty:
            return pd.DataFrame()

        if len(tickers) == 1:
            precios = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            precios = data["Close"]

        precios = precios.dropna(axis=1, how="all")
        return precios

    except Exception:
        return pd.DataFrame()


def _calcular_variacion(precios: pd.DataFrame, dias: int) -> pd.Series:
    """
    Calcula variación porcentual entre el precio de hace `dias` días
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


def _top_losers(variacion: pd.Series, n: int = TOP_N) -> pd.DataFrame:
    """Devuelve el ranking de las N mayores caídas (valores más negativos)."""
    if variacion.empty:
        return pd.DataFrame(columns=["Ticker", "Variación (%)"])

    losers = variacion.sort_values().head(n)
    df = losers.reset_index()
    df.columns = ["Ticker", "Variación (%)"]
    df["Variación (%)"] = df["Variación (%)"].round(2)
    return df


def _top_gainers(variacion: pd.Series, n: int = TOP_N) -> pd.DataFrame:
    """Devuelve el ranking de las N mayores subidas (valores más positivos)."""
    if variacion.empty:
        return pd.DataFrame(columns=["Ticker", "Variación (%)"])

    gainers = variacion.sort_values(ascending=False).head(n)
    df = gainers.reset_index()
    df.columns = ["Ticker", "Variación (%)"]
    df["Variación (%)"] = df["Variación (%)"].round(2)
    return df


# ─────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class MarketScanner:
    """
    Escanea los mercados ByMA y NYSE/NASDAQ buscando las mayores caídas
    o las mayores subidas en distintos períodos de tiempo.
    """

    def __init__(self, mercado: str = "nyse"):
        """
        Parameters
        ----------
        mercado : str — 'byma' o 'nyse'
        """
        assert mercado in ("byma", "nyse"), "mercado debe ser 'byma' o 'nyse'"
        self.mercado = mercado
        self.tickers = BYMA_TICKERS if mercado == "byma" else NYSE_TICKERS

    def escanear(self, dias: int, modo: str = "losers", progress_callback=None) -> pd.DataFrame:
        """
        Descarga datos y calcula el top de mayores caídas o subidas para el período dado.

        Parameters
        ----------
        dias : int  — cantidad de días hacia atrás (30, 90, 180, 365)
        modo : str  — 'losers' para mayores caídas, 'gainers' para mayores subidas
        progress_callback : callable opcional — recibe (0.0–1.0)

        Returns
        -------
        pd.DataFrame con columnas ['Ticker', 'Variación (%)']
        """
        precios = _descargar_precios(self.tickers, dias)

        if progress_callback:
            progress_callback(0.7)

        variacion = _calcular_variacion(precios, dias)

        if progress_callback:
            progress_callback(1.0)

        if modo == "gainers":
            return _top_gainers(variacion)
        return _top_losers(variacion)

    def escanear_todos_periodos(self, modo: str = "losers", progress_callback=None) -> dict:
        """
        Ejecuta el escaneo para todos los períodos definidos en PERIODOS.

        Parameters
        ----------
        modo : str — 'losers' para mayores caídas, 'gainers' para mayores subidas

        Returns
        -------
        dict { 'nombre_periodo': pd.DataFrame }
        """
        # Descargamos 1 año de datos de una sola vez (cubre todos los períodos)
        precios = _descargar_precios(self.tickers, 365)

        rank_fn = _top_gainers if modo == "gainers" else _top_losers

        resultados = {}
        periodos_items = list(PERIODOS.items())

        for i, (nombre, dias) in enumerate(periodos_items):
            variacion = _calcular_variacion(precios, dias)
            resultados[nombre] = rank_fn(variacion)

            if progress_callback:
                progress_callback((i + 1) / len(periodos_items))

        return resultados
