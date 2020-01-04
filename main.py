import sys
import numpy as np
import logging
from SystolicArray import SystolicArray
import os
import time


def main():

    basename = os.path.basename(__file__)[:-3]

    logging.basicConfig(level=logging.DEBUG,
                        filemode='w',
                        filename='{}.log'.format(basename),
                        format="[%(levelname)s] - %(asctime)s - %(name)s - %(message)s",
                        datefmt='%H:%M:%S')

    MainLogger = logging.getLogger('MainLogger')
    loggingNow = False

    MainLogger.info('Welcome to Multithreaded Systolic Array Experiment. By the captain we want to wish you good flight.')

    thread_count: int    = 2                         # [1, inf].
    array_size: int      = 16                        # [1, inf]. PE's array size - most be squared.
    probability_for_zero = 0.3                       # [0,1].    Probability to have Zero in a cell.
    buffer_depth         = array_size-2

    top_value            = 10
    values               = np.arange(top_value)      # Matrices values would rand from: [0,top_value]

    # We consider specified probability for 'zero' and Uniform Distribution on the rest.
    probabilities        = [probability_for_zero] + [(1-probability_for_zero)/values[1:].shape[0] for v in values[1:]]

    MainLogger.info('Basic Configuration:'
                    '\n\t1) Number of Threads: {}'
                    '\n\t2) Systolic Array size: {}x{}'
                    '\n\t3) Buffer depth: {}'
                    '\n\t4) Values: {}'
                    '\n\t5) With probabilities: {}'.format(thread_count, array_size, array_size, buffer_depth, str(values), str(probabilities)))

    # Multiple data and weights matrices - one per each thread
    data_matrices   = np.random.choice(values, (thread_count, array_size, array_size*100), p=probabilities)
    weight_matrices = np.random.choice(values, (thread_count, array_size*100, array_size), p=probabilities)

    MainLogger.info('Inputs Matrices:\n-------------------------------------------------'
                    '\nWest Matrices Shape (data):\n---------------------------\n{}'
                    '\nWest Matrices (data):\n---------------------\n{}'
                    '\nNorth Matrices Shape (weights):\n-------------------------------\n{}'
                    '\nNorth Matrices (weights):\n-------------------------\n{}'.format(data_matrices.shape, data_matrices, weight_matrices.shape, weight_matrices))

    # Algebraic matrices multiplication results
    # From numpy reference:
    # ---------------------
    # If either argument is N-D, N > 2,
    # it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    result_matrices = np.matmul(data_matrices, weight_matrices)

    MainLogger.info('Expected Result Matrices:\n----------------------------------------------------------\n{}'.format(result_matrices))

    MainLogger.info('Create Systolic Array Object')

    # MTSA C'tor
    systolic_array = SystolicArray(west_matrices=data_matrices,
                                   north_matrices=weight_matrices,
                                   array_size=array_size,
                                   thread_count=thread_count,
                                   buffer_depth=buffer_depth,
                                   log=loggingNow)

    # Each iteration is a clock cycle
    while 1:

        systolic_array.tick(log=loggingNow)

        # Check if done
        if systolic_array.isDone():

            # Gather some data
            systolic_array.summarize()
            break

    # Check for correctness
    if np.any(systolic_array.results - result_matrices):

        MainLogger.error("Systolic Array Provided False Results")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    sys.exit()
