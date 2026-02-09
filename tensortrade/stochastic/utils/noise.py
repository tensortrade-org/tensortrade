# Copyright 2020 The TensorTrade Authors.
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

"""Stochastic noise generators implemented with numpy."""

import numpy as np


class GaussianNoise:
    """Gaussian noise generator.

    Generates samples from a Gaussian (normal) distribution.

    Parameters
    ----------
    t : float, default 1.0
        The time parameter (used for scaling).
    rng : np.random.Generator, optional
        Random number generator. If None, uses default numpy RNG.

    Examples
    --------
    >>> noise = GaussianNoise(t=100)
    >>> samples = noise.sample(100)
    >>> len(samples)
    100
    """

    def __init__(self, t: float = 1.0, rng: np.random.Generator | None = None):
        self.t = t
        self.rng = rng or np.random.default_rng()

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples of Gaussian noise.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of n Gaussian noise samples.
        """
        return self.rng.standard_normal(n)


class FractionalBrownianMotion:
    """Fractional Brownian Motion (fBM) generator.

    Generates samples from a fractional Brownian motion process using
    the Davies-Harte method with FFT for O(n log n) complexity.

    Parameters
    ----------
    t : float, default 1.0
        The time horizon.
    hurst : float, default 0.5
        The Hurst parameter (0 < hurst < 1).
        - hurst = 0.5: standard Brownian motion
        - hurst > 0.5: persistent/trending behavior
        - hurst < 0.5: anti-persistent/mean-reverting behavior
    rng : np.random.Generator, optional
        Random number generator. If None, uses default numpy RNG.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Fractional_Brownian_motion
    [2] Davies, R. B., & Harte, D. S. (1987). Tests for Hurst effect.

    Examples
    --------
    >>> fbm = FractionalBrownianMotion(t=100, hurst=0.7)
    >>> samples = fbm.sample(100)
    >>> len(samples)
    100
    """

    def __init__(
        self,
        t: float = 1.0,
        hurst: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        if not 0 < hurst < 1:
            raise ValueError("Hurst parameter must be between 0 and 1 (exclusive)")
        self.t = t
        self.hurst = hurst
        self.rng = rng or np.random.default_rng()

    def _autocovariance(self, k: np.ndarray, hurst: float) -> np.ndarray:
        """Compute the autocovariance function for fBM increments."""
        return 0.5 * (
            np.abs(k - 1) ** (2 * hurst)
            - 2 * np.abs(k) ** (2 * hurst)
            + np.abs(k + 1) ** (2 * hurst)
        )

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples of fractional Brownian motion.

        Uses the Davies-Harte method with FFT for efficient O(n log n) generation.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of n fBM samples.
        """
        # For very small n, use simple cumulative sum of correlated noise
        if n <= 1:
            return np.zeros(max(1, n))

        # Compute autocovariance for the first row of circulant matrix
        # We need 2n points for the circulant embedding
        m = 2 * n

        # Build the first row of the circulant matrix
        # r[k] for k = 0, 1, ..., n and r[2n-k] = r[k] for k = 1, ..., n-1
        r = np.zeros(m)
        r[: n + 1] = self._autocovariance(np.arange(n + 1), self.hurst)
        r[n + 1 :] = r[n - 1 : 0 : -1]  # Mirror for circulant structure

        # Eigenvalues via FFT (real since matrix is symmetric)
        eigenvalues = np.fft.fft(r).real

        # Check for negative eigenvalues (Davies-Harte condition)
        # If negative, fall back to approximate method
        if np.any(eigenvalues < 0):
            return self._sample_approximate(n)

        # Generate complex Gaussian in frequency domain
        # For real output, we need conjugate symmetry
        sqrt_eigenvalues = np.sqrt(eigenvalues / m)

        # Generate random phases
        z_real = self.rng.standard_normal(m)
        z_imag = self.rng.standard_normal(m)
        z = z_real + 1j * z_imag

        # Multiply by sqrt of eigenvalues
        w = sqrt_eigenvalues * z

        # Inverse FFT to get samples
        samples = np.fft.ifft(w).real

        # Take first n samples as fBM increments
        increments = samples[:n]

        # Cumulative sum to get fBM path
        fbm_path = np.cumsum(increments)

        # Scale by time
        scale = (self.t / n) ** self.hurst
        return fbm_path * scale

    def _sample_approximate(self, n: int) -> np.ndarray:
        """Approximate fBM using Hosking's sequential method.

        Fallback when Davies-Harte eigenvalues are negative.
        """
        H = self.hurst

        # Generate standard normal increments
        gn = self.rng.standard_normal(n)

        # Initialize fBM increments
        fgn = np.zeros(n)
        fgn[0] = gn[0]

        if n == 1:
            return fgn * (self.t**H)

        # Hosking's method for generating fGn
        phi = np.zeros(n)
        psi = np.zeros(n)
        cov = self._autocovariance(np.arange(n), H)

        # First step
        v = cov[0]
        phi[0] = 0

        for i in range(1, n):
            # Update phi using Durbin-Levinson
            phi[i - 1] = cov[i]
            for j in range(i - 1):
                phi[i - 1] -= psi[j] * cov[i - j - 1]
            phi[i - 1] /= v

            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]

            v *= 1 - phi[i - 1] ** 2
            psi[:i] = phi[:i]

            # Generate fGn
            fgn[i] = np.sqrt(v) * gn[i]
            for j in range(i):
                fgn[i] += psi[j] * fgn[i - j - 1]

        # Cumulative sum to get fBM
        fbm_path = np.cumsum(fgn)

        # Scale by time
        scale = (self.t / n) ** H
        return fbm_path * scale
