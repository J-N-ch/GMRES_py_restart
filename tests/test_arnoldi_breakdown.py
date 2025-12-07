import sys, os
import numpy as np
import pytest
import warnings

# add library paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GMRES_API'))
import GMRES

def test_arnoldi_breakdown_no_warning():
    # Setup a simple system where solution is reached quickly
    A = np.eye(3)
    b = np.array([1.0, 1.0, 1.0])

    # Use max basis > dimension to force breakdown
    gmres = GMRES.GMRES_API(A, b, 5, 1e-16)
    gmres.initial_guess_input(np.zeros(3))

    # This should not emit RuntimeWarning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always") # Cause all warnings to always be triggered.
        x = gmres.run()

        # Filter out warnings that are not related to the fix if any (but we expect none)
        # Specifically we want to ensure no "invalid value encountered in divide"
        relevant_warnings = [w for w in record if "invalid value encountered in divide" in str(w.message)]
        assert len(relevant_warnings) == 0

    assert np.allclose(x, b)
    # Check that we don't have NaNs
    assert not np.isnan(x).any()

def test_arnoldi_breakdown_logic():
    # Direct test of arnoldi static method
    A = np.eye(3)
    # Basis vector that is already an eigenvector (so A*v is parallel to v)
    Q = np.zeros((3, 2))
    v = np.array([1.0, 0.0, 0.0])
    Q[:, 0] = v
    k = 0

    # A * v = v.
    # Arnoldi: q = A*v = v.
    # h[0] = v . v = 1.
    # q = v - 1*v = 0.
    # h[1] = norm(0) = 0.
    # q = q / 0 -> Should be handled.

    h, q = GMRES.GMRES_API.arnoldi(A, Q, k)

    assert h[1] == 0.0
    assert not np.isnan(q).any()
    assert not np.isinf(q).any()
    assert np.allclose(q, np.zeros(3))
