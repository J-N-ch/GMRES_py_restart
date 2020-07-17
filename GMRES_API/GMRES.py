import scipy.linalg as splinalg

import numpy as np
import math

#test_Givens_out_of_for_loop = False
test_Givens_out_of_for_loop = True

class GMRES_API(object):
    def __init__( self,
                  A_coefficient_matrix: np.array([], dtype = float ),
                  b_boundary_condition_vector: np.array([], dtype = float ),
                  maximum_number_of_basis_used: int,
                  threshold = 1.0e-16 ):

        self.A = A_coefficient_matrix
        self.b = b_boundary_condition_vector
        self.maximum_number_of_basis_used = maximum_number_of_basis_used
        self.threshold = threshold

    def initial_guess_input( self, x_input_vector_initial_guess: np.array([], dtype = float ) ):

        self.x = x_input_vector_initial_guess

        try:
            assert len( self.x ) == len( self.b )

        except Exception:

            print(" The input guess vector's size must equal to the system's size !\n")
            print(" The matrix system's size == ", len( self.b ))
            print(" Your input vector's size == ", len( self.x ))
            self.x = np.zeros( len( self.b ) ) 
            print(" Use default input guess vector = ", self.x, " instead of the incorrect vector you given !\n")


    def run( self ):

        n = len( self.A )
        m = self.maximum_number_of_basis_used

        r = self.b - np.dot(self.A , self.x)
        r_norm = np.linalg.norm( r )

        b_norm = np.linalg.norm( self.b )

        self.error = np.linalg.norm( r ) / b_norm
        self.e = [self.error]
        
        # initialize the 1D vectors 
        sn = np.zeros( m )
        cs = np.zeros( m )
        e1 = np.zeros( m + 1 )
        e1[0] = 1.0

        # beta is the beta vector instead of the beta scalar
        beta = r_norm * e1 
        beta_test = r_norm * e1


        H = np.zeros(( m+1, m+1 ))
        H_test = np.zeros(( m+1, m+1 ))

        Q = np.zeros((   n, m+1 ))
        Q[:,0] = r / r_norm

        #---------------------------------------------------------------------------------------------
        for k in range(m):

            ( H[0:k+2, k], Q[:, k+1] )    = __class__.arnoldi( self.A, Q, k)
            #H_test[:,k] = H[:,k]
            H_test = H
            print("H_test =\n",H_test)
            if test_Givens_out_of_for_loop is not True:
                ( H[0:k+2, k], cs[k], sn[k] ) = __class__.apply_givens_rotation( H[0:k+2, k], cs, sn, k)
                # update the residual vector
                beta[ k+1 ] = -sn[k] * beta[k]
                beta[ k   ] =  cs[k] * beta[k]

                # calculate and save the errors
                self.error = abs(beta[k+1]) / b_norm
                self.e = np.append(self.e, self.error)

                if( self.error <= self.threshold):
                    break
        #---------------------------------------------------------------------------------------------


        if test_Givens_out_of_for_loop is True:
                
            # 1. My first GMRES written using Givens rotation to solve lstsq
            #---------------------------------------------------------------------------------------------------------------------
            H_Givens_test = np.copy(H_test)
            for k in range(m):
                ( H_Givens_test[0:k+2, k], cs[k], sn[k] ) = __class__.apply_givens_rotation( H_Givens_test[0:k+2, k], cs, sn, k)
                # update the residual vector
                beta[ k+1 ] = -sn[k] * beta[k]
                beta[ k   ] =  cs[k] * beta[k]
            print("H_Givens_test =\n", H_Givens_test)
            print("beta =\n", beta)
            #y = __class__.__back_substitution( H_Givens_test[0:k+1, 0:k+1], beta[0:k+1] )
            #y = np.matmul( np.linalg.inv( H_Givens_test[0:k+1, 0:k+1]), beta[0:k+1] )
            #y = np.linalg.lstsq(H_Givens_test[0:m, 0:m], beta[0:m])[0]
            #y = splinalg.solve_triangular(H_Givens_test[0:m, 0:m],beta[0:m] )
            #---------------------------------------------------------------------------------------------------------------------

            # 2. GMRES using QR decomposition to solve lstsq
            #-----------------------------------------------------
            H_QR_test = np.copy(H_test)
            QR_q, QR_r = np.linalg.qr(H_QR_test, mode='reduced')
            #print(QR_q)
            print("QR_r =\n", QR_r)
            #print(beta)
            new_beta = np.matmul(  QR_q.T, beta )
            #print(new_beta[0:m])
            print("new_beta =",new_beta)
            #y = np.linalg.lstsq(QR_r[0:m, 0:m],new_beta[0:m] )[0]
            #y = np.linalg.lstsq(QR_r[:,0:m],new_beta )[0]
            #y = splinalg.solve_triangular(QR_r[0:m, 0:m],new_beta[0:m] )
            #-----------------------------------------------------

            # 3. GMRES directly using numpy.linalg.lstsq  to solve lstsq (the most success one until now !)
            #-------------------------------------------------------------
            y = np.linalg.lstsq(H_test[0:m+1, 0:m], beta_test)[0]
            #y = np.linalg.solve(H_test[0:m, 0:m], beta_test[0:m])
            print(H_test[0:m+1, 0:m])
            print(beta_test)
            #print(np.linalg.solve(H_test[0:m, 0:m], beta_test[0:m]))
            #-------------------------------------------------------------

            #y = np.matmul( np.linalg.inv( H_test[0:m, 0:m]), beta_test[0:m] )
            #y = np.linalg.solve( H_test[0:m, 0:m], beta_test[0:m] )


        else:
            # 1. My first GMRES written using Givens rotation to solve lstsq(but put the Givens with arnoldi)
            #-----------------------------------------------------------------------------------
            # calculate the result
            #y = np.matmul( np.linalg.inv( H[0:k+1, 0:k+1]), beta[0:k+1] )
            #TODO Due to H[0:k+1, 0:k+1] being a upper tri-matrix, we can exploit this fact. 
            y = __class__.__back_substitution( H[0:k+1, 0:k+1], beta[0:k+1] )
            #-----------------------------------------------------------------------------------


        #print("y =", y)
        self.x = self.x + np.matmul( Q[:,0:k+1], y )


        self.final_residual_norm = np.linalg.norm( self.b - np.matmul( self.A, self.x ) )

        return self.x


    '''''''''''''''''''''''''''''''''''
    '        Arnoldi Function         '
    '''''''''''''''''''''''''''''''''''
    @staticmethod
    def arnoldi( A, Q, k ):
        h = np.zeros( k+2 )
        q = np.dot( A, Q[:,k] )
        for i in range ( k+1 ):
            h[i] = np.dot( q, Q[:,i])
            q = q - h[i] * Q[:, i]
        h[ k+1 ] = np.linalg.norm(q)
        q = q / h[ k+1 ]
        return h, q 

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '           Applying Givens Rotation to H col           '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    @staticmethod
    def apply_givens_rotation( h, cs, sn, k ):
        for i in range( k-1 ):
            temp   =  cs[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
            h[i]   = temp

        # update the next sin cos values for rotation
        cs_k, sn_k, h[k] = __class__.givens_rotation( h[k-1], h[k] )
        
        # eliminate H[ k+1, i ]
        h[k + 1] = 0.0

        return h, cs_k, sn_k

    ##----Calculate the Given rotation matrix----##
    # From "http://www.netlib.org/lapack/lawnspdf/lawn150.pdf"
    # The algorithm used by "Edward Anderson"
    @staticmethod
    def givens_rotation( v1, v2 ):
        if( v2 == 0.0 ):
            cs = np.sign(v1)
            sn = 0.0
            r = abs(v1)
        elif( v1 == 0.0 ):
            cs = 0.0
            sn = np.sign(v2)
            r = abs(v2)
        elif( abs(v1) > abs(v2) ):
            t = v2 / v1 
            u = np.sign(v1) * math.hypot( 1.0, t )  
            cs = 1.0 / u
            sn = t * cs
            r = v1 * u
        else:
            t = v1 / v2 
            u = np.sign(v2) * math.hypot( 1.0, t )  
            sn = 1.0 / u
            cs = t * sn
            r = v2 * u
        return cs, sn, r

    # From https://stackoverflow.com/questions/47551069/back-substitution-in-python
    @staticmethod
    def __back_substitution( A: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = b.size
        if A[n-1, n-1] == 0.0:
            raise ValueError

        x = np.zeros_like(b)
        x[n-1] = b[n-1] / A[n-1, n-1]
        for i in range( n-2, -1, -1 ):
            bb = 0
            for j in range ( i+1, n ):
                bb += A[i, j] * x[j]
            x[i] = (b[i] - bb) / A[i, i]
        return x

    def final_residual_info_show( self ):
        print( "x  =", self.x, "residual_norm =  ", self.final_residual_norm ) 
    
def main():

    A_mat = np.array( [[1.00, 1.00, 1.00],
                       [1.00, 2.00, 1.00],
                       [0.00, 0.00, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0] )

    GMRES_itr2 = GMRES_API( A_mat, b_mat, 2, 0.01)

    x_mat = np.array( [1.0, 1.0, 1.0] )
    print("x  =", x_mat)

    # GMRES with restart, 2 iterations in each restart ( GMRES(2) )
    max_restart_counts = 100
    for restart_counter in range(max_restart_counts):
        GMRES_itr2.initial_guess_input( x_mat )

        x_mat = GMRES_itr2.run()
        print(restart_counter+1," : x  =", x_mat)

    xx = np.matmul( np.linalg.inv(A_mat), b_mat )
    print("ANS : xx =", xx) 


if __name__ == '__main__':
    main()
