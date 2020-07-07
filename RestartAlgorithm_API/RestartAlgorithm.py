import numpy as np

class RestartAlgorithm(object):
    def __init__( self, *args, **kwargs ):

        # The default number of restarting 
        self.max_rst_iter = 1

        # The default iteration's ending threshold 
        self.restarting_iteration_ending_threshold = 1.0e-16 

    def kernel_algorithm_register( self, kernel_algorithm ):
        self.k_algo = kernel_algorithm

    def restart_initial_input( self, initial_input_vector ):
        self.init_in = initial_input_vector

    def maximum_restarting_iteration_register( self, maximum_restarting_iteration ):
        self.max_rst_iter = maximum_restarting_iteration

    def restarting_iteration_ending_threshold_register( self, restarting_iteration_ending_threshold ):
        self.restarting_iteration_ending_threshold = restarting_iteration_ending_threshold 

    def run_restart( self ):
        self.restart_output = np.array([], dtype = float )
        self.final_residual_trend = np.array([], dtype = float )

        try:         
            for restart_counter in range(self.max_rst_iter):

                self.k_algo.initial_guess_input( self.init_in )

                # run the kernel algorithm in each restart
                self.restart_output = self.k_algo.run()
                self.init_in = self.restart_output

                print( restart_counter+1, ": ", end = '' )
                self.k_algo.final_residual_info_show() 
                self.final_residual_trend = np.append( self.final_residual_trend, self.k_algo.final_residual_norm )

                if( self.k_algo.final_residual_norm < self.restarting_iteration_ending_threshold ): 
                    print("\nThe restarting iteration's ending threshold ",self.restarting_iteration_ending_threshold," has been reached !\n")
                    break

        except:
            print("\n !! ERROR !! Some parameters stiil have not been registered !!!\n")
            if 'self.k_algo' not in locals():
                print(" Please use \"kernel_algorithm_register( <your_kernel_algorithm> )\" to register a kernel algorithm !!\n")

            if 'self.init_in' not in locals():
                print(" Please use \"restart_initial_input( <your_initial_input_vector> )\" to register a initial input-vector !!\n")

        finally:
            return self.restart_output, self.final_residual_trend

