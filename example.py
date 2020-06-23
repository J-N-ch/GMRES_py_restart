import numpy as np
import GMRES
import RestartAlgorithm
from matplotlib import pyplot as plt 

def main():

    A_mat = np.array( [[1.00, 1.00, 1.00],
                       [1.50, 2.00, 1.00],
                       [0.30, 0.50, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0] )

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2, 0.01)
    x_mat = np.array( [1.0, 1.0, 1.0] )
    print("x  =", x_mat)


    # The restatrt algorithm of GMRES
    #==============================================================================
    restart_number_one = RestartAlgorithm.RestartAlgorithm()
    restart_number_one.kernel_algorithm_register( GMRES_test_itr2 )
    restart_number_one.restart_initial_input( x_mat )
    restart_number_one.maximum_restarting_iteration_register( 200 )
    x_final, r_trend = restart_number_one.run_restart()
    #==============================================================================

    plt.title("restarted_GMRES_residual_trend") 
    plt.xlabel("restart") 
    plt.ylabel("residual") 
    plt.plot(r_trend)
    plt.show()


    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 

if __name__ == '__main__':
    main()

