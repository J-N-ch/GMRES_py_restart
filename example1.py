import numpy as np

import sys
sys.path.append('./GMRES_API')
import GMRES
sys.path.append('./RestartAlgorithm_API')
import RestartAlgorithm

from matplotlib import pyplot as plt 

def main():

    A_mat = np.array( [[4.00, 1.00, 1.00, 1.11],
                       [1.50, 3.00, 1.00, 1.73],
                       [1.50, 2.00, 3.00, 1.17],
                       [0.30, 0.50, 3.00, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0, 1.5] )



    # The phenomena described in 
    # M. Embree
    # The tortoise and the hare restart GMRES
    # SIAM Review, 45 (2) (2003), pp. 256-266 
    #=====================================================================================
    # GMRES with restart, 1 iterations in each restart ( GMRES(1) )
    #GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 1) # Converged, the fastest

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2) # Converged, but not the fastest

    # GMRES with restart, 3 iterations in each restart ( GMRES(3) )
    #GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 3) # This will explode

    # GMRES with restart, 4 iterations in each restart ( GMRES(4) )
    #GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 4) # This will explode
    #=====================================================================================



    x_mat = np.array( [1.0, 1.0, 1.0, 1.0] )
    print("x  =", x_mat)


    # The restatrt algorithm of GMRES
    #=============================================================
    restarted_GMRES = RestartAlgorithm.RestartAlgorithm()
    restarted_GMRES.kernel_algorithm_register( GMRES_test_itr2 )
    restarted_GMRES.restart_initial_input( x_mat )
    restarted_GMRES.maximum_restarting_iteration_register( 150 )
    x_final, r_trend = restarted_GMRES.run_restart()
    #=============================================================


    # Draw the residual trend by the sequence of restarts
    #============================================
    plt.title("restarted_GMRES_residual_trend") 
    plt.xlabel("restart") 
    plt.ylabel("residual") 
    plt.plot(r_trend)
    plt.show()
    #============================================


    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 

if __name__ == '__main__':
    main()


