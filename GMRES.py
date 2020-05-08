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
        print("x = ", self.x)

    def run( self ):
        print("GMRES run!")

        

    
def main():

    A_mat = np.array( [[0.01, 0.00],
                       [0.00, 0.01]] )

    b_mat = np.array( [1.00, 0.00] )

    GMRES_test_No1 = GMRES_API( A_mat, b_mat, 1, 0.01)

    x_mat = np.array( [0.01, 0.01] )

    GMRES_test_No1.initial_guess_input( x_mat )

    GMRES_test_No1.run()


if __name__ == '__main__':
    main()
