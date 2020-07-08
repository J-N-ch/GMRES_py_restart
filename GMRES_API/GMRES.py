import numpy as np

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
            assert len( self.x ) is len( self.b )

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

        beta = r_norm * e1 
        # beta is the beta vector instead of the beta scalar


        self.H = np.zeros( ( m+1, m+1 ) )
        self.Q = np.zeros( (   n, m+1 ) )
        self.Q[:,0] = r / r_norm
        self.Q_norm = np.linalg.norm( self.Q )

        
        for k in range(m):

            ( self.H[0:k+2, k], self.Q[:, k+1] ) = self.arnoldi( self.A, self.Q, k)

            ( self.H[0:k+2, k], cs[k], sn[k] ) = self.apply_givens_rotation(self.H[0:k+2, k], cs, sn, k)
            
            # update the residual vector
            beta[k+1] = -sn[k] * beta[k]
            beta[k]   =  cs[k] * beta[k]

            # calcilate the error
            self.error = abs(beta[k+1]) / b_norm
            
            # save the error
            self.e = np.append(self.e, self.error)

            if( self.error <= self.threshold):
                break

        # calculate the result
        #TODO Due to self.H[0:k+1, 0:k+1] being a upper tri-matrix, we can exploit this fact. 
        self.y = self.__back_substitution( self.H[0:k+1, 0:k+1], beta[0:k+1] )
        #self.y = np.matmul( np.linalg.inv(self.H[0:k+1, 0:k+1]), beta[0:k+1] )

        self.x = self.x + np.matmul(self.Q[:,0:k+1], self.y)

        self.final_residual_norm = np.linalg.norm( self.b - np.matmul( self.A, self.x ) )

        return self.x


    '''''''''''''''''''''''''''''''''''
    '        Arnoldi Function         '
    '''''''''''''''''''''''''''''''''''
    def arnoldi( self, A, Q, k):
        h = np.zeros( k+2 )
        q = np.dot( A, Q[:,k] )
        for i in range (k+1):
            h[i] = np.dot( q, Q[:,i])
            q = q - h[i] * Q[:, i]
        h[k + 1] = np.linalg.norm(q)
        q = q / h[k + 1]
        return h, q 

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '           Applying Givens Rotation to H col           '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def apply_givens_rotation( self, h, cs, sn, k ):
        for i in range( k-1 ):
            temp   =  cs[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
            h[i]   = temp
        # update the next sin cos values for rotation
        [ cs_k, sn_k ] = self.givens_rotation( h[k-1], h[k] )
        
        # eliminate H[ i + 1, i ]
        h[k] = cs_k * h[k] + sn_k * h[k + 1]
        h[k + 1] = 0.0
        return h, cs_k, sn_k

    ##----Calculate the Given rotation matrix----##
    def givens_rotation( self, v1, v2 ):
        if(v1 == 0):
            cs = 0
            sn = 1
        else:
            t = np.sqrt(v1**2 + v2**2)
            cs = abs(v1) / t
            sn = cs * v2 / v1
        return cs, sn

    # From https://stackoverflow.com/questions/47551069/back-substitution-in-python
    def __back_substitution(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = b.size
        x = np.zeros_like(b)

        if A[n-1, n-1] == 0:
            raise ValueError

        x[n-1] = b[n-1]/A[n-1, n-1]
        C = np.zeros((n,n))
        for i in range(n-2, -1, -1):
            bb = 0
            for j in range (i+1, n):
                bb += A[i, j]*x[j]

            C[i, i] = b[i] - bb
            x[i] = C[i, i]/A[i, i]

        return x

    def final_residual_info_show( self ):
        print("x  =", self.x, "residual_norm =  ", self.final_residual_norm ) 
    
