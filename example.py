
import numpy as np

import sys
sys.path.append('./GMRES_API')
import GMRES
sys.path.append('./RestartAlgorithm_API')
import RestartAlgorithm

from matplotlib import pyplot as plt 

def run_GMRES_restart( methods_used_to_solve_leastSqare, A_mat, b_mat, x_mat ):
    # The restatrt algorithm of GMRES
    #=====================================================================================================
    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2)
    GMRES_test_itr2.methods_used_to_solve_leastSqare_register( methods_used_to_solve_leastSqare )
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr2 )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 22 )
    restarted_GMRES.restarting_iteration_ending_threshold_register( 1.0e-14 )
    x_final, r_trend = restarted_GMRES.run_restart()
    #=====================================================================================================
    return x_final, r_trend

def main():

    A_mat = np.array( [
                       [1.00, 1.00, 1.00],
                       [1.50, 2.00, 1.00],
                       [0.30, 0.50, 3.00],
                      ] )

    b_mat = np.array( [
                       3.0,
                       2.0,
                       1.0,
                      ] )

    x_mat = np.array( [
                       1.0,
                       1.0,
                       1.0,
                      ] )
    print("x  =", x_mat)


    # The algorithm of GMRES without using restart
    #============================================================================================================================
    size_of_matrix_A = len( A_mat )
    number_of_orthogonal_basis_to_be_constructed = size_of_matrix_A 
    original_GMRES_test = GMRES.GMRES_API( A_mat, b_mat, number_of_orthogonal_basis_to_be_constructed )
    original_GMRES_test.initial_guess_input( x_mat )
    original_GMRES_test.methods_used_to_solve_leastSqare_register("leastSquare_solver_numpy")
    original_GMRES_final_x = original_GMRES_test.run()
    print("original_GMRES_final_x = ", original_GMRES_final_x, "residual_norm = ", original_GMRES_test.final_residual_norm,"\n")
    #============================================================================================================================


    """
    # The restatrt algorithm of GMRES
    #======================================================================================
    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2)
    #GMRES_test_itr2.methods_used_to_solve_leastSqare_register("Givens_rotation")
    #GMRES_test_itr2.methods_used_to_solve_leastSqare_register("QR_decomposition_numpy")
    GMRES_test_itr2.methods_used_to_solve_leastSqare_register("leastSquare_solver_numpy")
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr2 )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 22 )
    restarted_GMRES.restarting_iteration_ending_threshold_register( 1.0e-14 )
    x_final, r_trend = restarted_GMRES.run_restart()
    #======================================================================================
    """
    x_final_1, r_trend_1 = run_GMRES_restart("Givens_rotation",          A_mat, b_mat, x_mat )
    x_final_2, r_trend_2 = run_GMRES_restart("QR_decomposition_numpy",   A_mat, b_mat, x_mat )
    x_final_3, r_trend_3 = run_GMRES_restart("leastSquare_solver_numpy", A_mat, b_mat, x_mat )

    print("  original_GMRES_final_x = ", original_GMRES_final_x, "residual_norm = ", original_GMRES_test.final_residual_norm)
    #print("restarting_GMRES_final_x = ", x_final, "residual_norm = ", GMRES_test_itr2.final_residual_norm)
    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("Ans directly solved :  x = ", xx) 


    # Draw the residual trend by the sequence of restarts
    #============================================
    plt.title("restarted_GMRES_residual_trend") 
    plt.xlabel("restart") 
    plt.ylabel("residual") 
    #plt.plot(r_trend)
    plt.plot(r_trend_1)
    plt.plot(r_trend_2)
    plt.plot(r_trend_3)
    plt.show()
    #============================================


if __name__ == '__main__':
    main()

