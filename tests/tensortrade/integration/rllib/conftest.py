# Copyright 2025 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Shared fixtures for RLlib integration tests."""

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n_samples = 200

    # Generate realistic price movement
    initial_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='h'),
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples),
    })

    # Ensure high >= close >= low and high >= open >= low
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def sample_csv_path(sample_ohlcv_data: pd.DataFrame, tmp_path: Path) -> str:
    """Write sample data to a temporary CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def minimal_env_config(sample_csv_path: str) -> dict:
    """Minimal environment configuration for testing."""
    return {
        "window_size": 5,
        "csv_filename": sample_csv_path,
        "max_allowed_loss": 0.5,
    }


@pytest.fixture(scope="module")
def ray_session() -> Generator[None, None, None]:
    """Initialize Ray session for module-scoped tests."""
    import ray

    ray.init(
        num_cpus=2,
        local_mode=True,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    yield
    ray.shutdown()
