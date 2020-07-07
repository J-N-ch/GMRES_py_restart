
import numpy as np

import sys
sys.path.append('./GMRES_API')
import GMRES
sys.path.append('./RestartAlgorithm_API')
import RestartAlgorithm

from matplotlib import pyplot as plt 


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
    #===================================================================================================
    size_of_matrix_A = len( A_mat )
    number_of_orthogonal_basis_to_be_constructed = size_of_matrix_A 
    original_GMRES_test = GMRES.GMRES_API( A_mat, b_mat, number_of_orthogonal_basis_to_be_constructed )
    original_GMRES_test.initial_guess_input( x_mat )
    original_GMRES_final_x = original_GMRES_test.run()
    print("original_GMRES_final_x = ", original_GMRES_final_x, "residual_norm = ", original_GMRES_test.final_residual_norm,"\n")
    #===================================================================================================


    # The restatrt algorithm of GMRES
    #=========================================================================
    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2)
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr2 )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 350 )
    restarted_GMRES.restarting_iteration_ending_threshold_register( 1.0e-15 )
    x_final, r_trend = restarted_GMRES.run_restart()
    #=========================================================================

    print("  original_GMRES_final_x = ", original_GMRES_final_x, "residual_norm = ", original_GMRES_test.final_residual_norm)
    print("restarting_GMRES_final_x = ", x_final, "residual_norm = ", GMRES_test_itr2.final_residual_norm)
    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("Ans directly solved :  x = ", xx) 


    # Draw the residual trend by the sequence of restarts
    #============================================
    plt.title("restarted_GMRES_residual_trend") 
    plt.xlabel("restart") 
    plt.ylabel("residual") 
    plt.plot(r_trend)
    plt.show()
    #============================================


if __name__ == '__main__':
    main()

