import sys, os
import numpy as np

# add library paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GMRES_API'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RestartAlgorithm_API'))

import GMRES
import RestartAlgorithm

def test_gmres_solution_matches_numpy():
    A = np.array([[1.0, 1.0, 1.0],
                  [1.5, 2.0, 1.0],
                  [0.3, 0.5, 3.0]])
    b = np.array([3.0, 2.0, 1.0])

    gmres = GMRES.GMRES_API(A, b, len(A))
    gmres.initial_guess_input(np.zeros(len(A)))
    gmres.methods_used_to_solve_leastSqare_register('leastSquare_solver_numpy')

    x = gmres.run()
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, atol=1e-8)
    assert gmres.final_residual_norm <= 1e-8

def test_restart_algorithm_converges():
    A = np.array([[1.0, 1.0, 1.0],
                  [1.5, 2.0, 1.0],
                  [0.3, 0.5, 3.0]])
    b = np.array([3.0, 2.0, 1.0])

    kernel = GMRES.GMRES_API(A, b, 2)
    kernel.methods_used_to_solve_leastSqare_register('leastSquare_solver_numpy')
    restart = RestartAlgorithm.RestartAlgorithm()
    restart.kernel_algorithm_register(kernel)
    restart.restart_initial_input(np.zeros(len(A)))
    restart.maximum_restarting_iteration_register(20)
    restart.restarting_iteration_ending_threshold_register(1e-8)

    x, trend = restart.run_restart()
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, atol=1e-8)
    assert trend.size > 0 and trend[-1] < 1e-8

def test_back_substitution():
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    b = np.array([5.0, 6.0])
    x = GMRES.GMRES_API._GMRES_API__back_substitution(A, b)
    assert np.allclose(x, np.array([1.5, 2.0]))
