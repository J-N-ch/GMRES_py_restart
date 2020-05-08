import numpy as np

class GMRES_API:
    def __init__( self, A: np.array([], dtype = float ), b: np.array([], dtype = float ), max_iterations: int, threshold: float):
        print("Hello")
    
def main():

    A_mat = np.array( [[0.01, 0.00],
                       [0.00, 0.01]] )

    b_mat = np.array( [1.00, 0.00] )

    GMRES_test_No1 = GMRES_API( A_mat, b_mat, 1, 0.01)

if __name__ == '__main__':
    main()
