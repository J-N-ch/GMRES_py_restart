import numpy as np

class GMRES_API:
    def __init__( self,
                  A: np.array([], dtype = float ),
                  b: np.array([], dtype = float ),
                  max_iterations: int,
                  threshold: float ):

        print("Hello GMRES")
        self.A = A
        self.b = b
        self.max_iterations = max_iterations
        self.threshold = threshold

    def initial_guess_input( self, x: np.array([], dtype = float ) ):
        print("Hello GMRES init input")
        self.x = x


    def run( self ):
        print("GMRES run!")
        print("A = \n", self.A)
        self.n = int( np.sqrt(np.size( self.A )) )
        print("size of A = ", self.n)

        self.m = self.max_iterations
        print("maximum iterations = ", self.m)

        print("b = ", self.b)

        print("x = ", self.x)
        
        self.r = self.b - np.dot(self.A , self.x)
        print("r = ", self.r)

        self.b_norm = np.linalg.norm( self.b )
        print("b_norm = ", self.b_norm)

        self.error = np.linalg.norm( self.r ) / self.b_norm
        print("error = ", self.error )
        
        # initialize the 1D vectors 
        self.sn = np.zeros( self.m )
        self.cs = np.zeros( self.m )
        self.e1 = np.zeros( self.n + 1 )
        self.e1[0] = 1.0
        print("e1 = ", self.e1)

        self.e = [self.error]
        self.r_norm = np.linalg.norm( self.r )
        print("r_norm = ", self.r_norm)

        self.H = np.zeros((self.m+1, self.n))
        print("H = ", self.H)
        self.Q = np.zeros((self.n, self.m+1))
        self.Q[:,0] = self.r / self.r_norm
        print("Q = ", self.Q)
        self.Q_norm = np.linalg.norm( self.Q )
        print("Q_norm = ", self.Q_norm)

        self.beta = self.r_norm * self.e1 
        # beta is the beta vector instead of the beta scalar
        print("beta = ", self.beta)
        
        for k in range(self.m):
            print(k)

            #print( self.arnoldi( self.A, self.Q, k) )
            ( self.H[0:k+2, k], self.Q[:, k+1] ) = self.arnoldi( self.A, self.Q, k)
            print(self.H)
            ( self.H[0:k+2, k], self.cs[k], self.sn[k] ) = self.apply_givens_rotation(self.H[0:k+2, k], self.cs, self.sn, k)
            print(self.H)
            
            # update the residual vector
            self.beta[k+1] = -self.sn[k] * self.beta[k]
            self.beta[k]   =  self.cs[k] * self.beta[k]
            self.error          = abs(self.beta[k+1]) / self.b_norm


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

    '''''''''''''''''''''''''''''''''''''''''''''
    '     Applying Givens Rotation to H col     '
    '''''''''''''''''''''''''''''''''''''''''''''
    def apply_givens_rotation( self, h, cs, sn, k ):
        print("k =", k)
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

    
def main():

    A_mat = np.array( [[1.00, 1.00, 1.00],
                       [1.00, 2.00, 1.00],
                       [0.00, 0.00, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0] )

    GMRES_test_No1 = GMRES_API( A_mat, b_mat, 3, 0.01)

    x_mat = np.array( [0.00, 0.10, 0.00] )

    GMRES_test_No1.initial_guess_input( x_mat )

    GMRES_test_No1.run()


if __name__ == '__main__':
    main()
