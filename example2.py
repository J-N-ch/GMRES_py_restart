import numpy as np

import sys
sys.path.append('./GMRES_API')
import GMRES
sys.path.append('./RestartAlgorithm_API')
import RestartAlgorithm

from matplotlib import pyplot as plt 

def GMRES_test( methods_used_to_solve_leastSqare, A_mat, b_mat, x_mat, iterations_between_restarts,restarting_iteration_ending_threshold ):
    print("\n")

    # The restatrt algorithm of GMRES
    #=========================================================================================================
    GMRES_test_itr = GMRES.GMRES_API( A_mat, b_mat, iterations_between_restarts )
    GMRES_test_itr.methods_used_to_solve_leastSqare_register( methods_used_to_solve_leastSqare )
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 150 )
    restarted_GMRES.restarting_iteration_ending_threshold_register( restarting_iteration_ending_threshold )
    x_final, r_trend = restarted_GMRES.run_restart()
    return  x_final, r_trend
    #=========================================================================================================



def main():

    A_mat = np.array( [[4.00, 1.00, 1.00, 1.11],
                       [1.50, 3.00, 1.00, 1.73],
                       [1.50, 2.00, 3.00, 1.17],
                       [0.30, 0.50, 3.00, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0, 1.5] )

    x_mat = np.array( [1.0, 1.0, 1.0, 1.0] )
    print("x  =", x_mat)

    method = "Givens_rotation" 
    #method = "QR_decomposition_numpy" 
    #method = "leastSquare_solver_numpy"

    #===============================================================
    # GMRES with restart, 1 iterations in each restart ( GMRES(1) )
    x_final_1, r_trend_1 = GMRES_test( method, A_mat, b_mat, x_mat, 1, 1.0e-15 )

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    x_final_2, r_trend_2 = GMRES_test( method, A_mat, b_mat, x_mat, 2, 1.0e-15 )

    # GMRES with restart, 3 iterations in each restart ( GMRES(3) )
    x_final_3, r_trend_3 = GMRES_test( method, A_mat, b_mat, x_mat, 3, 1.0e-15 )

    # GMRES with restart, 4 iterations in each restart ( GMRES(4) )
    #x_final_4, r_trend_4 = GMRES_test( method, A_mat, b_mat, x_mat, 4, 1.0e-15 )
    #===============================================================

    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 

    # Draw the residual trend by the sequence of restarts
    #============================================
    plt.title("restarted_GMRES_residual_trend") 
    plt.xlabel("restart") 
    plt.ylabel("residual") 
    max_restart_shown = 150
    plt.plot(r_trend_1[0:max_restart_shown])
    plt.plot(r_trend_2[0:max_restart_shown])
    plt.plot(r_trend_3[0:max_restart_shown])
    #plt.plot(r_trend_4[0:max_restart_shown])
    plt.show()
    #============================================



if __name__ == '__main__':
    main()


