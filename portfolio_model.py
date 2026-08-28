"""
Portfolio Analyzer Model
========================
Lógica de análisis de portafolio de inversión usando datos reales
descargados desde Yahoo Finance con yfinance.

Autor: Curso Python
Versión: 2.0 — Catálogo ampliado (ByMA + NYSE/NASDAQ completo)
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────
# CATÁLOGO DE ACTIVOS DISPONIBLES
# ─────────────────────────────────────────────────────────────────

ACTIVOS = {
    # ── Mercado Argentino (ByMA) ─────────────────────────────────
    "A3 (ByMA)":            "A3.BA",
    "A3D (ByMA)":           "A3D.BA",
    "AGRO (ByMA)":          "AGRO.BA",
    "AGROD (ByMA)":         "AGROD.BA",
    "ALUA (ByMA)":          "ALUA.BA",
    "ALUAD (ByMA)":         "ALUAD.BA",
    "AUSO (ByMA)":          "AUSO.BA",
    "BBAR (ByMA)":          "BBAR.BA",
    "BHIP (ByMA)":          "BHIP.BA",
    "BHIPD (ByMA)":         "BHIPD.BA",
    "BMA (ByMA)":           "BMA.BA",
    "BOLT (ByMA)":          "BOLT.BA",
    "BPAT (ByMA)":          "BPAT.BA",
    "BYMA (ByMA)":          "BYMA.BA",
    "CADO (ByMA)":          "CADO.BA",
    "CAPX (ByMA)":          "CAPX.BA",
    "CARC (ByMA)":          "CARC.BA",
    "CECO2 (ByMA)":         "CECO2.BA",
    "CELU (ByMA)":          "CELU.BA",
    "CEPU (ByMA)":          "CEPU.BA",
    "CEPUD (ByMA)":         "CEPUD.BA",
    "CGPA2 (ByMA)":         "CGPA2.BA",
    "COME (ByMA)":          "COME.BA",
    "CRES (ByMA)":          "CRES.BA",
    "CTIO (ByMA)":          "CTIO.BA",
    "CVH (ByMA)":           "CVH.BA",
    "DGCE (ByMA)":          "DGCE.BA",
    "DGCU2 (ByMA)":         "DGCU2.BA",
    "ECOG (ByMA)":          "ECOG.BA",
    "ECOGD (ByMA)":         "ECOGD.BA",
    "EDN (ByMA)":           "EDN.BA",
    "EDND (ByMA)":          "EDND.BA",
    "FERR (ByMA)":          "FERR.BA",
    "FIPL (ByMA)":          "FIPL.BA",
    "GAMI (ByMA)":          "GAMI.BA",
    "GARO (ByMA)":          "GARO.BA",
    "GBAN (ByMA)":          "GBAN.BA",
    "GCDI (ByMA)":          "GCDI.BA",
    "GCLA (ByMA)":          "GCLA.BA",
    "GGAL (ByMA)":          "GGAL.BA",
    "GGALD (ByMA)":         "GGALD.BA",
    "HARG (ByMA)":          "HARG.BA",
    "HAVA (ByMA)":          "HAVA.BA",
    "HSAT (ByMA)":          "HSAT.BA",
    "IEB (ByMA)":           "IEB.BA",
    "INTR (ByMA)":          "INTR.BA",
    "INVJ (ByMA)":          "INVJ.BA",
    "IRS2W (ByMA)":         "IRS2W.BA",
    "IRSA (ByMA)":          "IRSA.BA",
    "IRSAD (ByMA)":         "IRSAD.BA",
    "LEDE (ByMA)":          "LEDE.BA",
    "LOMA (ByMA)":          "LOMA.BA",
    "LONG (ByMA)":          "LONG.BA",
    "METR (ByMA)":          "METR.BA",
    "MIRG (ByMA)":          "MIRG.BA",
    "MOLA (ByMA)":          "MOLA.BA",
    "MOLI (ByMA)":          "MOLI.BA",
    "MORI (ByMA)":          "MORI.BA",
    "OEST (ByMA)":          "OEST.BA",
    "PAMP (ByMA)":          "PAMP.BA",
    "PAMPD (ByMA)":         "PAMPD.BA",
    "PATA (ByMA)":          "PATA.BA",
    "RAGH (ByMA)":          "RAGH.BA",
    "RICH (ByMA)":          "RICH.BA",
    "RIGO (ByMA)":          "RIGO.BA",
    "ROSE (ByMA)":          "ROSE.BA",
    "SAMI (ByMA)":          "SAMI.BA",
    "SEMI (ByMA)":          "SEMI.BA",
    "SUPV (ByMA)":          "SUPV.BA",
    "TECO2 (ByMA)":         "TECO2.BA",
    "TGN4D (ByMA)":         "TGN4D.BA",
    "TGSU2 (ByMA)":         "TGSU2.BA",
    "TRAN (ByMA)":          "TRAN.BA",
    "TXAR (ByMA)":          "TXAR.BA",
    "VALO (ByMA)":          "VALO.BA",
    "YPFD (ByMA)":          "YPFD.BA",
    "YPFDD (ByMA)":         "YPFDD.BA",
    # ── NYSE / NASDAQ ────────────────────────────────────────────
    "AAL":                  "AAL",
    "AAP":                  "AAP",
    "Apple (AAPL)":         "AAPL",
    "ABBV":                 "ABBV",
    "ABEV":                 "ABEV",
    "ABNB":                 "ABNB",
    "ABT":                  "ABT",
    "ACN":                  "ACN",
    "ACWI":                 "ACWI",
    "Adobe (ADBE)":         "ADBE",
    "ADP":                  "ADP",
    "AEM":                  "AEM",
    "AI":                   "AI",
    "AIG":                  "AIG",
    "ALAB":                 "ALAB",
    "AMAT":                 "AMAT",
    "AMD":                  "AMD",
    "AMX":                  "AMX",
    "Amazon (AMZN)":        "AMZN",
    "ARKK":                 "ARKK",
    "ARM":                  "ARM",
    "ASML":                 "ASML",
    "ASTS":                 "ASTS",
    "Broadcom (AVGO)":      "AVGO",
    "AXP":                  "AXP",
    "AZN":                  "AZN",
    "Boeing (BA)":          "BA",
    "Alibaba (BABA)":       "BABA",
    "BAK":                  "BAK",
    "BB":                   "BB",
    "BBD":                  "BBD",
    "Baidu (BIDU)":         "BIDU",
    "BIIB":                 "BIIB",
    "BIOX":                 "BIOX",
    "BITF":                 "BITF",
    "BK":                   "BK",
    "BKNG":                 "BKNG",
    "BKR":                  "BKR",
    "BMNR":                 "BMNR",
    "Berkshire (BRKB)":     "BRKB",
    "BX":                   "BX",
    "Citigroup (C)":        "C",
    "CAAP":                 "CAAP",
    "CAH":                  "CAH",
    "CAT":                  "CAT",
    "CDE":                  "CDE",
    "CEG":                  "CEG",
    "CIBR":                 "CIBR",
    "CL":                   "CL",
    "CLS":                  "CLS",
    "Coinbase (COIN)":      "COIN",
    "COPX":                 "COPX",
    "Costco (COST)":        "COST",
    "Salesforce (CRM)":     "CRM",
    "CRWV":                 "CRWV",
    "Cisco (CSCO)":         "CSCO",
    "CVS":                  "CVS",
    "Chevron (CVX)":        "CVX",
    "CX":                   "CX",
    "Delta (DAL)":          "DAL",
    "DD":                   "DD",
    "John Deere (DE)":      "DE",
    "DECK":                 "DECK",
    "DEO":                  "DEO",
    "DHR":                  "DHR",
    "DIA":                  "DIA",
    "DOCU":                 "DOCU",
    "DOW":                  "DOW",
    "EA":                   "EA",
    "eBay (EBAY)":          "EBAY",
    "ECL":                  "ECL",
    "EEM":                  "EEM",
    "EFA":                  "EFA",
    "EFX":                  "EFX",
    "EQNR":                 "EQNR",
    "ESGU":                 "ESGU",
    "ETHA":                 "ETHA",
    "ETSY":                 "ETSY",
    "EWZ":                  "EWZ",
    "Ford (F)":             "F",
    "FDX":                  "FDX",
    "FNMA":                 "FNMA",
    "FSLR":                 "FSLR",
    "FXI":                  "FXI",
    "GDX":                  "GDX",
    "GE":                   "GE",
    "GFI":                  "GFI",
    "Gilead (GILD)":        "GILD",
    "GLD":                  "GLD",
    "Globant (GLOB)":       "GLOB",
    "GLW":                  "GLW",
    "GM":                   "GM",
    "Google (GOOGL)":       "GOOGL",
    "GPRK":                 "GPRK",
    "GRMN":                 "GRMN",
    "Goldman (GS)":         "GS",
    "GT":                   "GT",
    "HAL":                  "HAL",
    "Home Depot (HD)":      "HD",
    "HL":                   "HL",
    "HMC":                  "HMC",
    "HMY":                  "HMY",
    "HOG":                  "HOG",
    "HON":                  "HON",
    "HOOD":                 "HOOD",
    "HPQ":                  "HPQ",
    "HSBC":                 "HSBC",
    "HSY":                  "HSY",
    "HUT":                  "HUT",
    "HWM":                  "HWM",
    "IBB":                  "IBB",
    "IBIT":                 "IBIT",
    "IBM":                  "IBM",
    "IBN":                  "IBN",
    "IEMG":                 "IEMG",
    "IEUR":                 "IEUR",
    "IFF":                  "IFF",
    "ILF":                  "ILF",
    "INFY":                 "INFY",
    "Intel (INTC)":         "INTC",
    "IP":                   "IP",
    "IREN":                 "IREN",
    "ISRG":                 "ISRG",
    "ITA":                  "ITA",
    "Itau (ITUB)":          "ITUB",
    "IVE":                  "IVE",
    "IVV":                  "IVV",
    "IVW":                  "IVW",
    "IWM":                  "IWM",
    "JD":                   "JD",
    "JMIA":                 "JMIA",
    "J&J (JNJ)":            "JNJ",
    "JOYY":                 "JOYY",
    "JPMorgan (JPM)":       "JPM",
    "KEP":                  "KEP",
    "KMB":                  "KMB",
    "Coca-Cola (KO)":       "KO",
    "LAC":                  "LAC",
    "LAR":                  "LAR",
    "Eli Lilly (LLY)":      "LLY",
    "LMT":                  "LMT",
    "LRCX":                 "LRCX",
    "LVS":                  "LVS",
    "Mastercard (MA)":      "MA",
    "McDonald's (MCD)":     "MCD",
    "MDLZ":                 "MDLZ",
    "MDT":                  "MDT",
    "MercadoLibre (MELI)":  "MELI",
    "Meta (META)":          "META",
    "MFG":                  "MFG",
    "MMM":                  "MMM",
    "MO":                   "MO",
    "MOS":                  "MOS",
    "Merck (MRK)":          "MRK",
    "Moderna (MRNA)":       "MRNA",
    "MRVL":                 "MRVL",
    "Microsoft (MSFT)":     "MSFT",
    "MSI":                  "MSI",
    "MicroStrategy (MSTR)": "MSTR",
    "MU":                   "MU",
    "MUFG":                 "MUFG",
    "MUX":                  "MUX",
    "NEM":                  "NEM",
    "Netflix (NFLX)":       "NFLX",
    "NG":                   "NG",
    "NIO":                  "NIO",
    "Nike (NKE)":           "NKE",
    "ServiceNow (NOW)":     "NOW",
    "NTES":                 "NTES",
    "Nu Holdings (NU)":     "NU",
    "NUE":                  "NUE",
    "NVIDIA (NVDA)":        "NVDA",
    "NVS":                  "NVS",
    "NXE":                  "NXE",
    "OKLO":                 "OKLO",
    "Oracle (ORCL)":        "ORCL",
    "ORLY":                 "ORLY",
    "OXY":                  "OXY",
    "PAAS":                 "PAAS",
    "PAGS":                 "PAGS",
    "PANW":                 "PANW",
    "PATH":                 "PATH",
    "Petrobras (PBR)":      "PBR",
    "PCAR":                 "PCAR",
    "PDD":                  "PDD",
    "PepsiCo (PEP)":        "PEP",
    "Pfizer (PFE)":         "PFE",
    "P&G (PG)":             "PG",
    "PINS":                 "PINS",
    "PKS":                  "PKS",
    "Palantir (PLTR)":      "PLTR",
    "PM":                   "PM",
    "PSQ":                  "PSQ",
    "PSX":                  "PSX",
    "PayPal (PYPL)":        "PYPL",
    "Qualcomm (QCOM)":      "QCOM",
    "QQQ":                  "QQQ",
    "Ferrari (RACE)":       "RACE",
    "Roblox (RBLX)":        "RBLX",
    "RGTI":                 "RGTI",
    "RIO":                  "RIO",
    "Riot (RIOT)":          "RIOT",
    "RKLB":                 "RKLB",
    "Roku (ROKU)":          "ROKU",
    "RTX":                  "RTX",
    "SAN":                  "SAN",
    "SAP":                  "SAP",
    "SATL":                 "SATL",
    "Starbucks (SBUX)":     "SBUX",
    "SCCO":                 "SCCO",
    "Sea Ltd (SE)":         "SE",
    "SH":                   "SH",
    "Shell (SHEL)":         "SHEL",
    "Shopify (SHOP)":       "SHOP",
    "SID":                  "SID",
    "SIEGY":                "SIEGY",
    "SLB":                  "SLB",
    "SLV":                  "SLV",
    "SMH":                  "SMH",
    "Snap (SNAP)":          "SNAP",
    "Snowflake (SNOW)":     "SNOW",
    "Sony (SONY)":          "SONY",
    "SPCE":                 "SPCE",
    "SPGI":                 "SPGI",
    "SPHQ":                 "SPHQ",
    "Spotify (SPOT)":       "SPOT",
    "SPXL":                 "SPXL",
    "SPY":                  "SPY",
    "STLA":                 "STLA",
    "STNE":                 "STNE",
    "SUZ":                  "SUZ",
    "AT&T (T)":             "T",
    "TCOM":                 "TCOM",
    "TD":                   "TD",
    "Atlassian (TEAM)":     "TEAM",
    "TEM":                  "TEM",
    "TGT":                  "TGT",
    "TJX":                  "TJX",
    "Toyota (TM)":          "TM",
    "TMUS":                 "TMUS",
    "TQQQ":                 "TQQQ",
    "TRIP":                 "TRIP",
    "Tesla (TSLA)":         "TSLA",
    "TSMC (TSM)":           "TSM",
    "TTE":                  "TTE",
    "Twilio (TWLO)":        "TWLO",
    "United Airlines (UAL)": "UAL",
    "Uber (UBER)":          "UBER",
    "Unilever (UL)":        "UL",
    "UnitedHealth (UNH)":   "UNH",
    "Union Pacific (UNP)":  "UNP",
    "UPST":                 "UPST",
    "URA":                  "URA",
    "URBN":                 "URBN",
    "USB":                  "USB",
    "USO":                  "USO",
    "Visa (V)":             "V",
    "Vale (VALE)":          "VALE",
    "VEA":                  "VEA",
    "VIG":                  "VIG",
    "Vista Energy (VIST)":  "VIST",
    "VRSN":                 "VRSN",
    "VST":                  "VST",
    "VXX":                  "VXX",
    "Verizon (VZ)":         "VZ",
    "Wells Fargo (WFC)":    "WFC",
    "Walmart (WMT)":        "WMT",
    "XLB":                  "XLB",
    "XLC":                  "XLC",
    "XLE":                  "XLE",
    "XLF":                  "XLF",
    "XLI":                  "XLI",
    "XLK":                  "XLK",
    "XLP":                  "XLP",
    "XLRE":                 "XLRE",
    "XLU":                  "XLU",
    "XLV":                  "XLV",
    "XLY":                  "XLY",
    "Exxon (XOM)":          "XOM",
    "XP":                   "XP",
    "XPEV":                 "XPEV",
    "Zoom (ZM)":            "ZM",
}


class PortfolioAnalyzer:
    """
    Descarga precios históricos y calcula métricas de portafolio.
    """

    def __init__(self, tickers: list, periodo: str = "1y"):
        """
        Parameters
        ----------
        tickers : list  — lista de símbolos (ej: ['AAPL', 'MELI'])
        periodo : str   — período de datos ('6mo', '1y', '2y', '5y')
        """
        self.tickers = tickers
        self.periodo = periodo
        self.precios  = None
        self.retornos = None
        self._descargar()

    # ── Descarga de datos ────────────────────────────────────────
    def _descargar(self):
        """Descarga precios de cierre ajustados desde Yahoo Finance"""
        data = yf.download(
            self.tickers,
            period=self.periodo,
            auto_adjust=True,
            progress=False,
        )

        # Manejar tanto 1 ticker como múltiples
        if len(self.tickers) == 1:
            self.precios = data[["Close"]].rename(columns={"Close": self.tickers[0]})
        else:
            self.precios = data["Close"]

        # Eliminar columnas totalmente vacías, y forward-fill para resolver
        # el desalineamiento de calendarios cuando se mezclan tickers de
        # distintos mercados (ej. ByMA + NYSE). OJO: no usar dropna() por
        # fila acá — con calendarios distintos borraría casi todas las fechas.
        self.precios  = self.precios.dropna(axis=1, how="all").ffill().dropna()
        self.retornos = self.precios.pct_change().dropna()

    # ── Métricas individuales ────────────────────────────────────
    def metricas_individuales(self) -> pd.DataFrame:
        """Retorno anualizado y volatilidad anualizada por activo"""
        retorno_anual = self.retornos.mean() * 252
        volatilidad   = self.retornos.std() * np.sqrt(252)
        sharpe        = retorno_anual / volatilidad

        df = pd.DataFrame({
            "Retorno Anual (%)":    (retorno_anual * 100).round(2),
            "Volatilidad Anual (%)": (volatilidad  * 100).round(2),
            "Sharpe Ratio":          sharpe.round(2),
        })
        return df

    # ── Métricas del portafolio ──────────────────────────────────
    def analizar_portafolio(self, pesos: dict) -> dict:
        """
        Calcula métricas del portafolio dado un diccionario de pesos.

        Parameters
        ----------
        pesos : dict — {ticker: peso_decimal}  (deben sumar 1)

        Returns
        -------
        dict con retorno, volatilidad, sharpe y evolución del portafolio
        """
        tickers_validos = [t for t in pesos if t in self.retornos.columns]
        w = np.array([pesos[t] for t in tickers_validos])

        # Normalizar por si no suman exactamente 1
        w = w / w.sum()

        ret = self.retornos[tickers_validos]

        # Retorno y volatilidad anualizados
        retorno_port = float((ret.mean() @ w) * 252)
        cov_matrix   = ret.cov() * 252
        volatilidad  = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe       = retorno_port / volatilidad if volatilidad > 0 else 0

        # Evolución acumulada del portafolio (base 100)
        retornos_port = (ret * w).sum(axis=1)
        evolucion     = (1 + retornos_port).cumprod() * 100

        # Máximo drawdown
        rolling_max  = evolucion.cummax()
        drawdown     = (evolucion - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min() * 100)

        return {
            "retorno_anual":   round(retorno_port * 100, 2),
            "volatilidad":     round(volatilidad  * 100, 2),
            "sharpe":          round(sharpe, 2),
            "max_drawdown":    round(max_drawdown, 2),
            "evolucion":       evolucion,
            "fechas":          evolucion.index,
        }

    # ── Correlación ──────────────────────────────────────────────
    def matriz_correlacion(self) -> pd.DataFrame:
        """Devuelve la matriz de correlación entre activos"""
        return self.retornos.corr().round(2)
