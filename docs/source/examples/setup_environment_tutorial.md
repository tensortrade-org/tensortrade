# Setup Environment


```python
import ta

import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
```

<br>**Fetch Historical Data**<br>

```python
cdd = CryptoDataDownload()

bitfinex_data = pd.concat([
    cdd.fetch("Bitfinex", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitfinex", "USD", "ETH", "1h").add_prefix("ETH:")
], axis=1)

bitstamp_data = pd.concat([
    cdd.fetch("Bitstamp", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitstamp", "USD", "LTC", "1h").add_prefix("LTC:")
], axis=1)
```


<br>**Define Exchanges**<br>

An exchange needs a name, an execution service, and streams of price data in order to function properly.

The setups supported right now are the simulated execution service using simulated or stochastic data. More execution services will be made available in the future, as well as price streams so that live data and execution can be supported.


```python
bitfinex = Exchange("bitfinex", service=execute_order)(
    Stream.source(list(bitfinex_data['BTC:close']), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitfinex_data['ETH:close']), dtype="float").rename("USD-ETH")
)

bitstamp = Exchange("bitstamp", service=execute_order)(
    Stream.source(list(bitstamp_data['BTC:close']), dtype="float").rename("USD-BTC"),
    Stream.source(list(bitstamp_data['LTC:close']), dtype="float").rename("USD-LTC")
)
```

Now that the exchanges have been defined we can define our features that we would like to include, excluding the prices we have provided for the exchanges.

<br>**Define External Data Feed**<br>

Here we will define the feed to use whatever data you would like. From financial indicators to alternative datasets, they will all have to be defined and incorporated into the `DataFeed` provided to the environment.


```python
# Add all features for bitstamp BTC & ETH
bitfinex_btc = bitfinex_data.loc[:, [name.startswith("BTC") for name in bitfinex_data.columns]]
bitfinex_eth = bitfinex_data.loc[:, [name.startswith("ETH") for name in bitfinex_data.columns]]

ta.add_all_ta_features(
    bitfinex_btc,
    colprefix="BTC:",
    **{k: "BTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
)


with NameSpace("bitfinex"):
    bitfinex_streams = [
        Stream.source(list(bitfinex_btc[c]), dtype="float").rename(c) for c in bitfinex_btc.columns
    ]
    bitfinex_streams += [
        Stream.source(list(bitfinex_eth[c]), dtype="float").rename(c) for c in bitfinex_eth.columns
    ]


# Add all features for bitstamp BTC & LTC
bitstamp_btc = bitstamp_data.loc[:, [name.startswith("BTC") for name in bitstamp_data.columns]]  
bitstamp_ltc = bitstamp_data.loc[:, [name.startswith("LTC") for name in bitstamp_data.columns]]

ta.add_all_ta_features(
    bitstamp_ltc,
    colprefix="LTC:",
    **{k: "LTC:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
)

with NameSpace("bitstamp"):
    bitstamp_streams = [
        Stream.source(list(bitstamp_btc[c]), dtype="float").rename(c) for c in bitstamp_btc.columns
    ]
    bitstamp_streams += [
        Stream.source(list(bitstamp_ltc[c]), dtype="float").rename(c) for c in bitstamp_ltc.columns
    ]


feed = DataFeed(bitfinex_streams + bitstamp_streams)
```


```python
feed.next()
```




    {'bitfinex:/BTC:date': Timestamp('2017-07-01 11:00:00'),
     'bitfinex:/BTC:open': 2505.56,
     'bitfinex:/BTC:high': 2513.38,
     'bitfinex:/BTC:low': 2495.12,
     'bitfinex:/BTC:close': 2509.17,
     'bitfinex:/BTC:volume': 287000.32,
     'bitfinex:/BTC:volume_adi': 462887.3781183644,
     'bitfinex:/BTC:volume_obv': nan,
     'bitfinex:/BTC:volume_cmf': 0.5388828039430464,
     'bitfinex:/BTC:volume_fi': nan,
     'bitfinex:/BTC:volume_em': nan,
     'bitfinex:/BTC:volume_vpt': -190920.02711825827,
     'bitfinex:/BTC:volume_nvi': 1000.0,
     'bitfinex:/BTC:volatility_atr': 85.51648155760596,
     'bitfinex:/BTC:volatility_bbh': 2509.17,
     'bitfinex:/BTC:volatility_bbl': 2509.17,
     'bitfinex:/BTC:volatility_bbm': 2509.17,
     'bitfinex:/BTC:volatility_bbhi': 0.0,
     'bitfinex:/BTC:volatility_bbli': 0.0,
     'bitfinex:/BTC:volatility_kcc': 2505.89,
     'bitfinex:/BTC:volatility_kch': 2524.15,
     'bitfinex:/BTC:volatility_kcl': 2487.6299999999997,
     'bitfinex:/BTC:volatility_kchi': 0.0,
     'bitfinex:/BTC:volatility_kcli': 0.0,
     'bitfinex:/BTC:volatility_dch': 2509.17,
     'bitfinex:/BTC:volatility_dcl': 2509.17,
     'bitfinex:/BTC:volatility_dchi': 0.0,
     'bitfinex:/BTC:volatility_dcli': 0.0,
     'bitfinex:/BTC:trend_macd': nan,
     'bitfinex:/BTC:trend_macd_signal': nan,
     'bitfinex:/BTC:trend_macd_diff': nan,
     'bitfinex:/BTC:trend_ema_fast': nan,
     'bitfinex:/BTC:trend_ema_slow': nan,
     'bitfinex:/BTC:trend_adx': 0.0,
     'bitfinex:/BTC:trend_adx_pos': 0.0,
     'bitfinex:/BTC:trend_adx_neg': 0.0,
     'bitfinex:/BTC:trend_vortex_ind_pos': nan,
     'bitfinex:/BTC:trend_vortex_ind_neg': nan,
     'bitfinex:/BTC:trend_vortex_diff': nan,
     'bitfinex:/BTC:trend_trix': nan,
     'bitfinex:/BTC:trend_mass_index': 0.0,
     'bitfinex:/BTC:trend_cci': nan,
     'bitfinex:/BTC:trend_dpo': 4963.073762705523,
     'bitfinex:/BTC:trend_kst': -664.2012654186367,
     'bitfinex:/BTC:trend_kst_sig': -664.2012654186367,
     'bitfinex:/BTC:trend_kst_diff': 0.0,
     'bitfinex:/BTC:trend_ichimoku_a': 2504.25,
     'bitfinex:/BTC:trend_ichimoku_b': 2504.25,
     'bitfinex:/BTC:trend_visual_ichimoku_a': 7460.129960014917,
     'bitfinex:/BTC:trend_visual_ichimoku_b': 7449.72498449202,
     'bitfinex:/BTC:trend_aroon_up': 4.0,
     'bitfinex:/BTC:trend_aroon_down': 4.0,
     'bitfinex:/BTC:trend_aroon_ind': 0.0,
     'bitfinex:/BTC:momentum_rsi': nan,
     'bitfinex:/BTC:momentum_mfi': nan,
     'bitfinex:/BTC:momentum_tsi': -100.0,
     'bitfinex:/BTC:momentum_uo': 0.2822915537138003,
     'bitfinex:/BTC:momentum_stoch': 76.94414019715232,
     'bitfinex:/BTC:momentum_stoch_signal': 76.94414019715232,
     'bitfinex:/BTC:momentum_wr': -23.055859802847678,
     'bitfinex:/BTC:momentum_ao': 0.0,
     'bitfinex:/BTC:momentum_kama': nan,
     'bitfinex:/BTC:others_dr': -66.42012654186367,
     'bitfinex:/BTC:others_dlr': nan,
     'bitfinex:/BTC:others_cr': 0.0,
     'bitfinex:/ETH:date': Timestamp('2017-07-01 11:00:00'),
     'bitfinex:/ETH:open': 279.98,
     'bitfinex:/ETH:high': 279.99,
     'bitfinex:/ETH:low': 272.1,
     'bitfinex:/ETH:close': 275.01,
     'bitfinex:/ETH:volume': 679358.87,
     'bitstamp:/BTC:date': Timestamp('2017-07-01 11:00:00'),
     'bitstamp:/BTC:open': 2506.5,
     'bitstamp:/BTC:high': 2510.62,
     'bitstamp:/BTC:low': 2495.5,
     'bitstamp:/BTC:close': 2500.0,
     'bitstamp:/BTC:volume': 521903.7,
     'bitstamp:/LTC:date': Timestamp('2017-07-01 11:00:00'),
     'bitstamp:/LTC:open': 39.67,
     'bitstamp:/LTC:high': 39.67,
     'bitstamp:/LTC:low': 39.32,
     'bitstamp:/LTC:close': 39.45,
     'bitstamp:/LTC:volume': 1957.48,
     'bitstamp:/LTC:volume_adi': 5133.608444728016,
     'bitstamp:/LTC:volume_obv': nan,
     'bitstamp:/LTC:volume_cmf': -0.2571428571428455,
     'bitstamp:/LTC:volume_fi': nan,
     'bitstamp:/LTC:volume_em': nan,
     'bitstamp:/LTC:volume_vpt': -895.4321955984681,
     'bitstamp:/LTC:volume_nvi': 1000.0,
     'bitstamp:/LTC:volatility_atr': 1.4838450097847482,
     'bitstamp:/LTC:volatility_bbh': 39.45,
     'bitstamp:/LTC:volatility_bbl': 39.45,
     'bitstamp:/LTC:volatility_bbm': 39.45,
     'bitstamp:/LTC:volatility_bbhi': 0.0,
     'bitstamp:/LTC:volatility_bbli': 0.0,
     'bitstamp:/LTC:volatility_kcc': 39.480000000000004,
     'bitstamp:/LTC:volatility_kch': 39.830000000000005,
     'bitstamp:/LTC:volatility_kcl': 39.13,
     'bitstamp:/LTC:volatility_kchi': 0.0,
     'bitstamp:/LTC:volatility_kcli': 0.0,
     'bitstamp:/LTC:volatility_dch': 39.45,
     'bitstamp:/LTC:volatility_dcl': 39.45,
     'bitstamp:/LTC:volatility_dchi': 0.0,
     'bitstamp:/LTC:volatility_dcli': 0.0,
     'bitstamp:/LTC:trend_macd': nan,
     'bitstamp:/LTC:trend_macd_signal': nan,
     'bitstamp:/LTC:trend_macd_diff': nan,
     'bitstamp:/LTC:trend_ema_fast': nan,
     'bitstamp:/LTC:trend_ema_slow': nan,
     'bitstamp:/LTC:trend_adx': 0.0,
     'bitstamp:/LTC:trend_adx_pos': 0.0,
     'bitstamp:/LTC:trend_adx_neg': 0.0,
     'bitstamp:/LTC:trend_vortex_ind_pos': nan,
     'bitstamp:/LTC:trend_vortex_ind_neg': nan,
     'bitstamp:/LTC:trend_vortex_diff': nan,
     'bitstamp:/LTC:trend_trix': nan,
     'bitstamp:/LTC:trend_mass_index': 0.0,
     'bitstamp:/LTC:trend_cci': nan,
     'bitstamp:/LTC:trend_dpo': 41.511479785526944,
     'bitstamp:/LTC:trend_kst': -512.7312383060929,
     'bitstamp:/LTC:trend_kst_sig': -512.7312383060929,
     'bitstamp:/LTC:trend_kst_diff': 0.0,
     'bitstamp:/LTC:trend_ichimoku_a': 39.495000000000005,
     'bitstamp:/LTC:trend_ichimoku_b': 39.495000000000005,
     'bitstamp:/LTC:trend_visual_ichimoku_a': 80.84515204884308,
     'bitstamp:/LTC:trend_visual_ichimoku_b': 80.77039939728148,
     'bitstamp:/LTC:trend_aroon_up': 4.0,
     'bitstamp:/LTC:trend_aroon_down': 4.0,
     'bitstamp:/LTC:trend_aroon_ind': 0.0,
     'bitstamp:/LTC:momentum_rsi': nan,
     'bitstamp:/LTC:momentum_mfi': nan,
     'bitstamp:/LTC:momentum_tsi': -100.0,
     'bitstamp:/LTC:momentum_uo': 0.31218871344045224,
     'bitstamp:/LTC:momentum_stoch': 37.14285714285772,
     'bitstamp:/LTC:momentum_stoch_signal': 37.14285714285772,
     'bitstamp:/LTC:momentum_wr': -62.85714285714228,
     'bitstamp:/LTC:momentum_ao': 0.0,
     'bitstamp:/LTC:momentum_kama': nan,
     'bitstamp:/LTC:others_dr': -51.27312383060929,
     'bitstamp:/LTC:others_dlr': nan,
     'bitstamp:/LTC:others_cr': 0.0}



<br>**Portfolio**<br>

Make the portfolio using the any combinations of exchanges and intruments that the exchange supports


```python
portfolio = Portfolio(USD, [
    Wallet(bitfinex, 10000 * USD),
    Wallet(bitfinex, 10 * BTC),
    Wallet(bitfinex, 5 * ETH),
    Wallet(bitstamp, 1000 * USD),
    Wallet(bitstamp, 5 * BTC),
    Wallet(bitstamp, 3 * LTC),
])
```

<br>**Environment**<br>


```python
env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="simple",
    feed=feed,
    window_size=15,
    enable_logger=False
)
```


```python
env.observer.feed.next()
```




    {'internal': {'bitfinex:/USD-BTC': 2509.17,
      'bitfinex:/USD-ETH': 275.01,
      'bitfinex:/USD:/free': 10000.0,
      'bitfinex:/USD:/locked': 0.0,
      'bitfinex:/USD:/total': 10000.0,
      'bitfinex:/BTC:/free': 10.0,
      'bitfinex:/BTC:/locked': 0.0,
      'bitfinex:/BTC:/total': 10.0,
      'bitfinex:/BTC:/worth': 25091.7,
      'bitfinex:/ETH:/free': 5.0,
      'bitfinex:/ETH:/locked': 0.0,
      'bitfinex:/ETH:/total': 5.0,
      'bitfinex:/ETH:/worth': 1375.05,
      'bitstamp:/USD-BTC': 2500.0,
      'bitstamp:/USD-LTC': 39.45,
      'bitstamp:/USD:/free': 1000.0,
      'bitstamp:/USD:/locked': 0.0,
      'bitstamp:/USD:/total': 1000.0,
      'bitstamp:/BTC:/free': 5.0,
      'bitstamp:/BTC:/locked': 0.0,
      'bitstamp:/BTC:/total': 5.0,
      'bitstamp:/BTC:/worth': 12500.0,
      'bitstamp:/LTC:/free': 3.0,
      'bitstamp:/LTC:/locked': 0.0,
      'bitstamp:/LTC:/total': 3.0,
      'bitstamp:/LTC:/worth': 118.35000000000001,
      'net_worth': 50085.1},
     'external': {'bitfinex:/BTC:date': Timestamp('2017-07-01 11:00:00'),
      'bitfinex:/BTC:open': 2505.56,
      'bitfinex:/BTC:high': 2513.38,
      'bitfinex:/BTC:low': 2495.12,
      'bitfinex:/BTC:close': 2509.17,
      'bitfinex:/BTC:volume': 287000.32,
      'bitfinex:/BTC:volume_adi': 462887.3781183644,
      'bitfinex:/BTC:volume_obv': nan,
      'bitfinex:/BTC:volume_cmf': 0.5388828039430464,
      'bitfinex:/BTC:volume_fi': nan,
      'bitfinex:/BTC:volume_em': nan,
      'bitfinex:/BTC:volume_vpt': -190920.02711825827,
      'bitfinex:/BTC:volume_nvi': 1000.0,
      'bitfinex:/BTC:volatility_atr': 85.51648155760596,
      'bitfinex:/BTC:volatility_bbh': 2509.17,
      'bitfinex:/BTC:volatility_bbl': 2509.17,
      'bitfinex:/BTC:volatility_bbm': 2509.17,
      'bitfinex:/BTC:volatility_bbhi': 0.0,
      'bitfinex:/BTC:volatility_bbli': 0.0,
      'bitfinex:/BTC:volatility_kcc': 2505.89,
      'bitfinex:/BTC:volatility_kch': 2524.15,
      'bitfinex:/BTC:volatility_kcl': 2487.6299999999997,
      'bitfinex:/BTC:volatility_kchi': 0.0,
      'bitfinex:/BTC:volatility_kcli': 0.0,
      'bitfinex:/BTC:volatility_dch': 2509.17,
      'bitfinex:/BTC:volatility_dcl': 2509.17,
      'bitfinex:/BTC:volatility_dchi': 0.0,
      'bitfinex:/BTC:volatility_dcli': 0.0,
      'bitfinex:/BTC:trend_macd': nan,
      'bitfinex:/BTC:trend_macd_signal': nan,
      'bitfinex:/BTC:trend_macd_diff': nan,
      'bitfinex:/BTC:trend_ema_fast': nan,
      'bitfinex:/BTC:trend_ema_slow': nan,
      'bitfinex:/BTC:trend_adx': 0.0,
      'bitfinex:/BTC:trend_adx_pos': 0.0,
      'bitfinex:/BTC:trend_adx_neg': 0.0,
      'bitfinex:/BTC:trend_vortex_ind_pos': nan,
      'bitfinex:/BTC:trend_vortex_ind_neg': nan,
      'bitfinex:/BTC:trend_vortex_diff': nan,
      'bitfinex:/BTC:trend_trix': nan,
      'bitfinex:/BTC:trend_mass_index': 0.0,
      'bitfinex:/BTC:trend_cci': nan,
      'bitfinex:/BTC:trend_dpo': 4963.073762705523,
      'bitfinex:/BTC:trend_kst': -664.2012654186367,
      'bitfinex:/BTC:trend_kst_sig': -664.2012654186367,
      'bitfinex:/BTC:trend_kst_diff': 0.0,
      'bitfinex:/BTC:trend_ichimoku_a': 2504.25,
      'bitfinex:/BTC:trend_ichimoku_b': 2504.25,
      'bitfinex:/BTC:trend_visual_ichimoku_a': 7460.129960014917,
      'bitfinex:/BTC:trend_visual_ichimoku_b': 7449.72498449202,
      'bitfinex:/BTC:trend_aroon_up': 4.0,
      'bitfinex:/BTC:trend_aroon_down': 4.0,
      'bitfinex:/BTC:trend_aroon_ind': 0.0,
      'bitfinex:/BTC:momentum_rsi': nan,
      'bitfinex:/BTC:momentum_mfi': nan,
      'bitfinex:/BTC:momentum_tsi': -100.0,
      'bitfinex:/BTC:momentum_uo': 0.2822915537138003,
      'bitfinex:/BTC:momentum_stoch': 76.94414019715232,
      'bitfinex:/BTC:momentum_stoch_signal': 76.94414019715232,
      'bitfinex:/BTC:momentum_wr': -23.055859802847678,
      'bitfinex:/BTC:momentum_ao': 0.0,
      'bitfinex:/BTC:momentum_kama': nan,
      'bitfinex:/BTC:others_dr': -66.42012654186367,
      'bitfinex:/BTC:others_dlr': nan,
      'bitfinex:/BTC:others_cr': 0.0,
      'bitfinex:/ETH:date': Timestamp('2017-07-01 11:00:00'),
      'bitfinex:/ETH:open': 279.98,
      'bitfinex:/ETH:high': 279.99,
      'bitfinex:/ETH:low': 272.1,
      'bitfinex:/ETH:close': 275.01,
      'bitfinex:/ETH:volume': 679358.87,
      'bitstamp:/BTC:date': Timestamp('2017-07-01 11:00:00'),
      'bitstamp:/BTC:open': 2506.5,
      'bitstamp:/BTC:high': 2510.62,
      'bitstamp:/BTC:low': 2495.5,
      'bitstamp:/BTC:close': 2500.0,
      'bitstamp:/BTC:volume': 521903.7,
      'bitstamp:/LTC:date': Timestamp('2017-07-01 11:00:00'),
      'bitstamp:/LTC:open': 39.67,
      'bitstamp:/LTC:high': 39.67,
      'bitstamp:/LTC:low': 39.32,
      'bitstamp:/LTC:close': 39.45,
      'bitstamp:/LTC:volume': 1957.48,
      'bitstamp:/LTC:volume_adi': 5133.608444728016,
      'bitstamp:/LTC:volume_obv': nan,
      'bitstamp:/LTC:volume_cmf': -0.2571428571428455,
      'bitstamp:/LTC:volume_fi': nan,
      'bitstamp:/LTC:volume_em': nan,
      'bitstamp:/LTC:volume_vpt': -895.4321955984681,
      'bitstamp:/LTC:volume_nvi': 1000.0,
      'bitstamp:/LTC:volatility_atr': 1.4838450097847482,
      'bitstamp:/LTC:volatility_bbh': 39.45,
      'bitstamp:/LTC:volatility_bbl': 39.45,
      'bitstamp:/LTC:volatility_bbm': 39.45,
      'bitstamp:/LTC:volatility_bbhi': 0.0,
      'bitstamp:/LTC:volatility_bbli': 0.0,
      'bitstamp:/LTC:volatility_kcc': 39.480000000000004,
      'bitstamp:/LTC:volatility_kch': 39.830000000000005,
      'bitstamp:/LTC:volatility_kcl': 39.13,
      'bitstamp:/LTC:volatility_kchi': 0.0,
      'bitstamp:/LTC:volatility_kcli': 0.0,
      'bitstamp:/LTC:volatility_dch': 39.45,
      'bitstamp:/LTC:volatility_dcl': 39.45,
      'bitstamp:/LTC:volatility_dchi': 0.0,
      'bitstamp:/LTC:volatility_dcli': 0.0,
      'bitstamp:/LTC:trend_macd': nan,
      'bitstamp:/LTC:trend_macd_signal': nan,
      'bitstamp:/LTC:trend_macd_diff': nan,
      'bitstamp:/LTC:trend_ema_fast': nan,
      'bitstamp:/LTC:trend_ema_slow': nan,
      'bitstamp:/LTC:trend_adx': 0.0,
      'bitstamp:/LTC:trend_adx_pos': 0.0,
      'bitstamp:/LTC:trend_adx_neg': 0.0,
      'bitstamp:/LTC:trend_vortex_ind_pos': nan,
      'bitstamp:/LTC:trend_vortex_ind_neg': nan,
      'bitstamp:/LTC:trend_vortex_diff': nan,
      'bitstamp:/LTC:trend_trix': nan,
      'bitstamp:/LTC:trend_mass_index': 0.0,
      'bitstamp:/LTC:trend_cci': nan,
      'bitstamp:/LTC:trend_dpo': 41.511479785526944,
      'bitstamp:/LTC:trend_kst': -512.7312383060929,
      'bitstamp:/LTC:trend_kst_sig': -512.7312383060929,
      'bitstamp:/LTC:trend_kst_diff': 0.0,
      'bitstamp:/LTC:trend_ichimoku_a': 39.495000000000005,
      'bitstamp:/LTC:trend_ichimoku_b': 39.495000000000005,
      'bitstamp:/LTC:trend_visual_ichimoku_a': 80.84515204884308,
      'bitstamp:/LTC:trend_visual_ichimoku_b': 80.77039939728148,
      'bitstamp:/LTC:trend_aroon_up': 4.0,
      'bitstamp:/LTC:trend_aroon_down': 4.0,
      'bitstamp:/LTC:trend_aroon_ind': 0.0,
      'bitstamp:/LTC:momentum_rsi': nan,
      'bitstamp:/LTC:momentum_mfi': nan,
      'bitstamp:/LTC:momentum_tsi': -100.0,
      'bitstamp:/LTC:momentum_uo': 0.31218871344045224,
      'bitstamp:/LTC:momentum_stoch': 37.14285714285772,
      'bitstamp:/LTC:momentum_stoch_signal': 37.14285714285772,
      'bitstamp:/LTC:momentum_wr': -62.85714285714228,
      'bitstamp:/LTC:momentum_ao': 0.0,
      'bitstamp:/LTC:momentum_kama': nan,
      'bitstamp:/LTC:others_dr': -51.27312383060929,
      'bitstamp:/LTC:others_dlr': nan,
      'bitstamp:/LTC:others_cr': 0.0}}
