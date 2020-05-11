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
        self.e1 = np.zeros( self.n )
        self.e1[0] = 1.0
        print("e1 = ", self.e1)

        self.e = [self.error]
        self.r_norm = np.linalg.norm( self.r )
        print("r_norm = ", self.r_norm)

        self.Q = np.zeros((self.n, self.m))
        self.Q[:,0] = self.r / self.r_norm
        print("Q = ", self.Q)
        self.Q_norm = np.linalg.norm( self.Q )
        print("Q_norm = ", self.Q_norm)

        self.beta = self.r_norm * self.e1 
        # beta is the beta vector instead of the beta scalar
        print("beta = ", self.beta)
        
        for k in range(0, self.m):
            print(k)
            print( self.arnoldi( self.A, self.Q, k) )

    def arnoldi( self, A, Q, k):
        print("Hello arnoldi working")
        h = np.zeros(k+1)
        q = np.dot(A, Q[:,k])
        for i in range (0, k):
            print( Q[:,i] ) 
            print(np.dot( q, Q[:,i]))
            h[i] = np.dot( q, Q[:,i])
            q = q - h[i] * Q[:, i]
        print("q = ", q)
        h[k] = np.linalg.norm(q)
        print("h = ", h)
        #q = q / h[k]

        return h, q 


    
def main():

    A_mat = np.array( [[1.00, 1.00, 1.00],
                       [0.00, 2.00, 1.00],
                       [0.00, 0.00, 3.00]] )

    b_mat = np.array( [3.0, 2.0, 1.0] )

    GMRES_test_No1 = GMRES_API( A_mat, b_mat, 3, 0.01)

    x_mat = np.array( [0.00, 0.10, 0.00] )

    GMRES_test_No1.initial_guess_input( x_mat )

    GMRES_test_No1.run()


if __name__ == '__main__':
    main()
