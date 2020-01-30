
from tensortrade.stochastic import fbm


def test_shape():
    frame = fbm(
        base_price=7000,
        base_volume=15000,
        start_date='2018-01-01',
        start_date_format='%Y-%m-%d',
        times_to_generate=1500,
        hurst=0.61,
        time_frame='1d'
    )

    assert frame.shape == (1500, 5)
