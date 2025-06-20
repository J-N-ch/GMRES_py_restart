import sys, os
import numpy as np

# add library paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GMRES_API'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RestartAlgorithm_API'))

import GMRES
import RestartAlgorithm


def test_restart_algorithm_stops_early():
    A = np.array([[4.0, 1.0],
                  [2.0, 3.0]])
    b = np.array([1.0, 1.0])

    kernel = GMRES.GMRES_API(A, b, 1)
    kernel.methods_used_to_solve_leastSqare_register('leastSquare_solver_numpy')
    restart = RestartAlgorithm.RestartAlgorithm()
    restart.kernel_algorithm_register(kernel)
    restart.restart_initial_input(np.zeros(len(A)))
    restart.maximum_restarting_iteration_register(5)
    restart.restarting_iteration_ending_threshold_register(1e-1)

    x, trend = restart.run_restart()

    assert len(trend) < 5
    assert trend[-1] < 1e-1
    assert np.allclose(np.dot(A, x), b, atol=1e-1)
