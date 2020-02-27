
import pandas as pd


class CryptoDataDownload:

    def __init__(self):
        self.url = "https://www.cryptodatadownload.com/cdd/"

    def fetch_default(self, exchange_name, base_symbol, quote_symbol, timeframe, include_all_volumes=False):

        filename = "{}_{}{}_{}.csv".format(exchange_name, quote_symbol, base_symbol, timeframe)
        base_vc = "Volume {}".format(base_symbol)
        new_base_vc = "volume_base"
        quote_vc = "Volume {}".format(quote_symbol)
        new_quote_vc = "volume_quote"

        df = pd.read_csv(self.url + filename, skiprows=1)
        df = df[::-1]
        df = df.drop(["Symbol"], axis=1)
        df = df.rename({base_vc: new_base_vc, quote_vc: new_quote_vc, "Date": "date"}, axis=1)

        if "d" in timeframe:
            df["date"] = pd.to_datetime(df["date"])
        elif "h" in timeframe:
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %I-%p")

        df = df.set_index("date")
        df.columns = [name.lower() for name in df.columns]
        df = df.reset_index()
        if not include_all_volumes:
            df = df.drop([new_quote_vc], axis=1)
            df = df.rename({new_base_vc: "volume"}, axis=1)
            return df
        return df

    def fetch_gemini(self, base_symbol, quote_symbol, timeframe):
        if timeframe.endswith("h"):
            timeframe = timeframe[:-1] + "hr"
        filename = "{}_{}{}_{}.csv".format("gemini", quote_symbol, base_symbol, timeframe)
        df = pd.read_csv(self.url + filename, skiprows=1)
        df = df[::-1]
        df = df.drop(["Symbol", "Unix Timestamp"], axis=1)
        df.columns = [name.lower() for name in df.columns]
        df = df.set_index("date")
        df = df.reset_index()
        return df

    def fetch(self, exchange_name, base_symbol, quote_symbol, timeframe, include_all_volumes=False):
        if exchange_name.lower() == "gemini":
            return self.fetch_gemini(base_symbol, quote_symbol, timeframe)
        return self.fetch_default(exchange_name, base_symbol, quote_symbol, timeframe, include_all_volumes=False)