import yfinance as yf


# one trip to yahoo finance. real prices only, zero made-up numbers
def download(tickers, period="2y"):
    raw = yf.download(tickers, period=period, auto_adjust=True, repair=True, progress=False)
    close = raw["Close"].dropna(how="all")  # auto_adjust = splits + dividends already baked in
    return close, close * raw["Volume"]     # dollar volume = price x shares traded


# price -> % move each day, only on days every stock traded
def returns(prices):
    return prices.dropna().pct_change().dropna()
