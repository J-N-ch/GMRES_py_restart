import numpy as np

class RestartAlgorithm(object):
    def __init__( self ):
        pass

    def kernel_algorithm_register( self, kernel_algorithm ):
        self.k_algo = kernel_algorithm
        #print( type( self.k_algo ) )

    def restart_initial_input( self, initial_input_vector ):
        self.init_in = initial_input_vector

    def maximum_restarting_iteration_register( self, maximum_restarting_iteration ):
        self.max_rst_iter = maximum_restarting_iteration
        #print(self.max_rst_iter)

    def run_restart( self ):
         
        self.final_residual_trend = np.array([], dtype = float )

        for restart_counter in range(self.max_rst_iter):

            self.k_algo.initial_guess_input( self.init_in )

            # run the kernel algorithm in each restart
            self.restart_output = self.k_algo.run()

            self.init_in = self.restart_output

            print( restart_counter+1, ": ", end = '' )

            self.final_residual_trend = np.append( self.final_residual_trend, self.k_algo.final_residual_info_show() )
             
        return self.restart_output, self.final_residual_trend

