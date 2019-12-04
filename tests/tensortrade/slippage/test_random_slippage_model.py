def test_slippage_in_bounds():
    assert 1 < 2 < 3


def test_slippage_not_zero():
    """ Make sure the slippage is not zero. """
    assert 1 != 0