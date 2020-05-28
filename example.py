import numpy as np
import GMRES

def main():

    A_mat = np.array( [[1.00, 1.00, 1.00],
                       [1.50, 2.00, 1.00],
                       [0.30, 0.50, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0] )

    GMRES_test_itr2 = GMRES.GMRES_API( A_mat, b_mat, 2, 0.01)

    x_mat = np.array( [1.0, 1.0, 1.0] )
    print("x  =", x_mat)

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    max_restart_counts = 200
    for restart_counter in range(max_restart_counts):
        GMRES_test_itr2.initial_guess_input( x_mat )

        x_mat = GMRES_test_itr2.run()

        residual_norm = np.linalg.norm( b_mat - np.matmul(A_mat, x_mat) )
        
        print( restart_counter+1," : x  =", x_mat, "residual_norm =  ", residual_norm )

    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 

if __name__ == '__main__':
    main()
