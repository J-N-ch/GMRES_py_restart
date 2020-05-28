import numpy as np

class GMRES_API:
    def __init__( self,
                  A: np.array([], dtype = float ),
                  b: np.array([], dtype = float ),
                  max_iterations: int,
                  threshold: float ):

        #print("Hello GMRES")
        self.A = A
        self.b = b
        self.max_iterations = max_iterations
        self.threshold = threshold

    def initial_guess_input( self, x: np.array([], dtype = float ) ):
        #print("Hello GMRES init input")
        self.x = x


    def run( self ):
        #print("GMRES run!")
        #print("A = \n", self.A)
        self.n = int( np.sqrt(np.size( self.A )) )
        #print("size of A = ", self.n)

        self.m = self.max_iterations
        #print("maximum iterations = ", self.m)

        #print("b = ", self.b)

        #print("x = ", self.x)
        
        self.r = self.b - np.dot(self.A , self.x)
        #print("r = ", self.r)

        self.b_norm = np.linalg.norm( self.b )
        #print("b_norm = ", self.b_norm)

        self.error = np.linalg.norm( self.r ) / self.b_norm
        #print("error = ", self.error )
        
        # initialize the 1D vectors 
        self.sn = np.zeros( self.m )
        self.cs = np.zeros( self.m )
        #self.e1 = np.zeros( self.n + 1 )
        self.e1 = np.zeros( self.m + 1 )
        self.e1[0] = 1.0
        #print("e1 = ", self.e1)

        self.e = [self.error]
        self.r_norm = np.linalg.norm( self.r )
        #print("r_norm = ", self.r_norm)

        self.H = np.zeros((self.m+1, self.m+1))
        #print("H = ", self.H)
        self.Q = np.zeros((self.n, self.m+1))
        self.Q[:,0] = self.r / self.r_norm
        #print("Q = ", self.Q)
        self.Q_norm = np.linalg.norm( self.Q )
        #print("Q_norm = ", self.Q_norm)

        self.beta = self.r_norm * self.e1 
        # beta is the beta vector instead of the beta scalar
        #print("beta = ", self.beta)
        
        for k in range(self.m):
            #print("k = ", k)

            #print( self.arnoldi( self.A, self.Q, k) )
            ( self.H[0:k+2, k], self.Q[:, k+1] ) = self.arnoldi( self.A, self.Q, k)
            #print(self.H)
            #print(self.Q)
            ( self.H[0:k+2, k], self.cs[k], self.sn[k] ) = self.apply_givens_rotation(self.H[0:k+2, k], self.cs, self.sn, k)
            #print(self.H)
            
            # update the residual vector
            self.beta[k+1] = -self.sn[k] * self.beta[k]
            self.beta[k]   =  self.cs[k] * self.beta[k]
            #print("beta = ",self.beta)
            self.error          = abs(self.beta[k+1]) / self.b_norm
            
            # save the error
            self.e = np.append(self.e, self.error)
            #print("k=",k,"error = ", self.error)
            #print("error = ", self.error)

            if( self.error <= self.threshold):
                break

        # calculate the result
        #TODO Due to self.H[0:k+1, 0:k+1] being a upper tri-matrix, we can exploit this fact. 
        self.y = self.__back_substitution( self.H[0:k+1, 0:k+1], self.beta[0:k+1] )
        #self.y = np.matmul( np.linalg.inv(self.H[0:k+1, 0:k+1]), self.beta[0:k+1] )

        self.x = self.x + np.matmul(self.Q[:,0:k+1], self.y)

        return self.x


    '''''''''''''''''''''''''''''''''''
    '        Arnoldi Function         '
    '''''''''''''''''''''''''''''''''''
    def arnoldi( self, A, Q, k):
        #print("k = ", k)
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
        #print("k =", k)
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
    
