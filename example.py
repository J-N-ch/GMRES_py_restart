import numpy as np
import GMRES
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

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2, 0.01)
    x_mat = np.array( [
                       1.0,
                       1.0,
                       1.0,
                      ] )
    print("x  =", x_mat)


    # The restatrt algorithm of GMRES
    #=============================================================
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr2 )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 35 )
    x_final, r_trend = restarted_GMRES.run_restart()
    #=============================================================

    print("x_final = ", x_final)
    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 


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

