"""Tests for shared Ray configuration helpers."""

from tensortrade.ray_config import build_ray_init_kwargs, resolve_ray_address


def test_resolve_ray_address_prefers_tensortrade_var(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setenv("TENSORTRADE_RAY_ADDRESS", "ray://cluster:10001")

    assert resolve_ray_address() == "ray://cluster:10001"


def test_build_ray_init_kwargs_local_defaults(monkeypatch):
    monkeypatch.delenv("TENSORTRADE_RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("TENSORTRADE_RAY_NAMESPACE", raising=False)

    kwargs = build_ray_init_kwargs(default_num_cpus=4)

    assert kwargs["ignore_reinit_error"] is True
    assert kwargs["log_to_driver"] is False
    assert kwargs["num_cpus"] == 4
    assert "address" not in kwargs


def test_build_ray_init_kwargs_shared_cluster(monkeypatch):
    monkeypatch.setenv("TENSORTRADE_RAY_ADDRESS", "auto")

    kwargs = build_ray_init_kwargs(default_num_cpus=8)

    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs
