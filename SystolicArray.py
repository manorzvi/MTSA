from PE        import PE, PElimited
from BUFFER    import BUFFER, OUTPUT, FIFO, BUFFERlimited
from Utilities import pack_FIFOs, unpack_BUFFERs
import numpy as np
import logging

SystolicArrayLogger = logging.getLogger('SystolicArrayLogger')


class SystolicArray:
    """
    Systolic Array Class.
    """

    def __init__(self, west_matrices, north_matrices, array_size, thread_count, buffer_depth, log):
        """
        Construct SystolicArray object.
        SystolicArray is the heart of our system.
        It consist of an array of PE's - atomic logic elements to multiply and add 2 scalars.
        Intermediate results stored in logical buffer between each PE.
        """

        if west_matrices.shape[0] != north_matrices.shape[0]:
            SystolicArrayLogger.critical("Threads number isn't equal in west matrix and north matrix")
            raise ValueError("Threads number isn't equal in west matrix and north matrix")
        if array_size != west_matrices.shape[1] or array_size != north_matrices.shape[2]:
            SystolicArrayLogger.critical("Systolic array size can't be different them matrices edges.")
            raise ValueError("Systolic array size can't be different them matrices edges.")
        if thread_count != west_matrices.shape[0]:
            SystolicArrayLogger.critical("Threads number isn't equal to west matrix threads")
            raise ValueError("Threads number isn't equal to west matrix threads")
        if thread_count != north_matrices.shape[0]:
            SystolicArrayLogger.critical("Threads number isn't equal to north matrix threads")
            raise ValueError("Threads number isn't equal to north matrix threads")
        if buffer_depth == 0 or buffer_depth == 1:
            SystolicArrayLogger.critical("Buffer Size most be at least 2.")
            raise ValueError("Buffer Size most be at least 2.")
        elif buffer_depth < 0:
            SystolicArrayLogger.debug("Unlimited Buffer Size")
        else:
            SystolicArrayLogger.debug("Limited Buffer Size: {}".format(buffer_depth))

        self.array_size   = array_size
        self.thread_count = thread_count

        if buffer_depth < 0:
            self.limited_buffer = False
        else:
            self.limited_buffer = True

        self.pe_array                = [] # PE's array
        self.horizontal_buffer_array = [] # Horizontal Buffers array
        self.vertical_buffer_array   = [] # Vertical Buffers array

        self.clock = 1
        SystolicArrayLogger.info("Clock: {}".format(self.clock))

        # Inputs from the west
        self.west_matrices        = west_matrices
        self.west_matrices_shape  = west_matrices.shape
        # Inputs from the north
        self.north_matrices       = north_matrices
        self.north_matrices_shape = north_matrices.shape

        self.results            = np.zeros((thread_count, array_size, array_size))
        self.utilization_per_pe = np.zeros((array_size, array_size))

        # Generate FIFO inputs objects. See docstring in pack_FIFOs function.
        if log:
            SystolicArrayLogger.debug("West Input Matrices:\n"
                                      "--------------------------------------------------------------")
        self.west_inputs  = pack_FIFOs(west_matrices, axis=0, thread_count=thread_count, log=log)

        if log:
            SystolicArrayLogger.debug("North Input Matrices:\n"
                                      "---------------------------------------------------------------")
        self.north_inputs = pack_FIFOs(north_matrices, axis=1, thread_count=thread_count, log=log)

        self.east_outputs  = []
        self.south_outputs = []

        # Generate PE's array.
        if log:
            SystolicArrayLogger.info('PE Array:\n'
                                     '---------------------------------------------------')
        for pe_iindex in range(array_size):
            self.pe_array.append([])

            for pe_jindex in range(array_size):

                if not self.limited_buffer:
                    self.pe_array[-1].append(PE(i=pe_iindex, j=pe_jindex, thread_count=thread_count, matrix_size=array_size, log=log))

                else:
                    self.pe_array[-1].append(PElimited(i=pe_iindex, j=pe_jindex, thread_count=thread_count, matrix_size=array_size, log=log))

        # Generate horizontal Buffers array.
        if log:
            SystolicArrayLogger.debug('Horizontal Buffers Array:\n'
                                      '-------------------------------------------------------------------')
        for b_iindex in range(array_size):
            self.horizontal_buffer_array.append([])

            for b_jindex in range(array_size):

                if b_jindex != array_size - 1:

                    # Different types of buffers for limited and unlimited buffers
                    if not self.limited_buffer:
                        self.horizontal_buffer_array[-1].append(BUFFER(thread_count=thread_count, iindex=b_iindex, jindex=b_jindex, log=log))

                    else:
                        self.horizontal_buffer_array[-1].append(BUFFERlimited(thread_count=thread_count, depth=buffer_depth, iindex=b_iindex, jindex=b_jindex, log=log))

                else:
                    self.horizontal_buffer_array[-1].append(OUTPUT(thread_count=thread_count, iindex=b_iindex, jindex=b_jindex, log=log))
                    self.east_outputs.append(self.horizontal_buffer_array[-1][-1])

        # Generate vertical Buffers array.
        if log:
            SystolicArrayLogger.debug('Vertical Buffers Array:\n'
                                      '-----------------------------------------------------------------')
        for b_iindex in range(array_size):
            self.vertical_buffer_array.append([])

            for b_jindex in range(array_size):

                if b_iindex != array_size - 1:

                    if not self.limited_buffer:
                        self.vertical_buffer_array[-1].append(BUFFER(thread_count=thread_count, iindex=b_iindex, jindex=b_jindex, log=log))

                    else:
                        self.vertical_buffer_array[-1].append(BUFFERlimited(thread_count=thread_count, depth=buffer_depth, iindex=b_iindex, jindex=b_jindex, log=log))

                else:
                    self.vertical_buffer_array[-1].append(OUTPUT(thread_count=thread_count, iindex=b_iindex, jindex=b_jindex, log=log))
                    self.south_outputs.append(self.vertical_buffer_array[-1][-1])

        # Connect PE's to adjacent Buffers
        if log:
            SystolicArrayLogger.debug("Connect PE's to Adjacent Buffers:\n"
                                      "---------------------------------------------------------------------------")
        for pe_iindex in range(array_size):

            for pe_jindex in range(array_size):

                if pe_jindex == 0 and pe_iindex == 0:  # Top corner PE. Interact with west inputs and north inputs
                    self.pe_array[0][0].connect(west_buffer=self.west_inputs[0],
                                                north_buffer=self.north_inputs[0],
                                                east_buffer=self.horizontal_buffer_array[0][0],
                                                south_buffer=self.vertical_buffer_array[0][0], log=log)

                elif pe_iindex == 0:  # West edge of PE's array. PE's interact with west inputs, and each other
                    self.pe_array[0][pe_jindex].connect(west_buffer=self.horizontal_buffer_array[0][pe_jindex - 1],
                                                        north_buffer=self.north_inputs[pe_jindex],
                                                        east_buffer=self.horizontal_buffer_array[0][pe_jindex],
                                                        south_buffer=self.vertical_buffer_array[0][pe_jindex], log=log)

                elif pe_jindex == 0:  # North edge of PE's array. PE's interact with north inputs, and each other
                    self.pe_array[pe_iindex][0].connect(west_buffer=self.west_inputs[pe_iindex],
                                                        north_buffer=self.vertical_buffer_array[pe_iindex - 1][0],
                                                        east_buffer=self.horizontal_buffer_array[pe_iindex][0],
                                                        south_buffer=self.vertical_buffer_array[pe_iindex][0], log=log)

                else:  # Inner PE's. Interact with each other.
                    self.pe_array[pe_iindex][pe_jindex].connect(
                        west_buffer=self.horizontal_buffer_array[pe_iindex][pe_jindex - 1],
                        north_buffer=self.vertical_buffer_array[pe_iindex - 1][pe_jindex],
                        east_buffer=self.horizontal_buffer_array[pe_iindex][pe_jindex],
                        south_buffer=self.vertical_buffer_array[pe_iindex][pe_jindex], log=log)

    def tick(self, log):
        """
        Single shift of data in between PE's
        :return: None
        """
        self.clock += 1

        SystolicArrayLogger.info("Raising Edge Clock: {}".format(self.clock))

        for pe_iindex in range(self.array_size):

            for pe_jindex in range(self.array_size):

                self.pe_array[pe_iindex][pe_jindex].tock(log=log)

        for buffer_iindex in range(self.array_size - 1):

            for buffer_jindex in range(self.array_size - 1):
                # Update Buffer's effective depth, for statistics extraction later on.
                self.horizontal_buffer_array[buffer_iindex][buffer_jindex].update_load()
                self.vertical_buffer_array[buffer_iindex][buffer_jindex].update_load()

    def isDone(self):
        """
        Check if matrix multiplication finished.
        Matrix multiplication finished if both west inputs and north inputs equals east outputs and south outputs
        :return: boolean. True for finished. False otherwise.
        """
        is_west_equal_east   = np.array_equal(self.west_matrices,  unpack_BUFFERs(buffer_list=self.east_outputs, axis=1,
                                                                                  matrix_shape=self.west_matrices_shape))
        is_north_equal_south = np.array_equal(self.north_matrices, unpack_BUFFERs(buffer_list=self.south_outputs, axis=0,
                                                                                  matrix_shape=self.north_matrices_shape))

        if is_west_equal_east and is_north_equal_south:

            SystolicArrayLogger.info("East Output Buffers Equal To West Input Matrices, South Output Buffers Equal To North Inputs Matrices. Systolic Array Done.")
            return True
        else:
            return False

    def summarize(self):
        """
        Summarize results.
        - reduce time to steady state off clock counter to overcome boundary values affection.
        - do the same for MAC progress utilization logger (count clock cycles on which MAC worked)
        - copy results from each PE to MTSA array.
        - calculate utilization per PE as the division result of total clock cycles by those on which each MAC worked.
        :return: None
        """

        # Time for steady-state of the Systolic Array: (<array_size>-1 * 2).
        # Therefore, we reduce that number*2 from clock counting (time to fill the Systolic Array, and time to evacuate)
        self.clock -= 2*((self.array_size-1)*2)

        # For the same reason, we delete from mac_utility list 2*(<array_size>-1) from the beginning,
        # and 2*(<array_size>-1) from the end
        for pe_iindex in range(self.array_size):

            for pe_jindex in range(self.array_size):

                self.pe_array[pe_iindex][pe_jindex].mac_utility = self.pe_array[pe_iindex][pe_jindex].mac_utility[2*((self.array_size-1)*2):]

                self.pe_array[pe_iindex][pe_jindex].mac_utility = self.pe_array[pe_iindex][pe_jindex].mac_utility[:-2*((self.array_size - 1)*2)]

        for pe_iindex in range(self.array_size):

            for pe_jindex in range(self.array_size):

                self.results[:, pe_iindex, pe_jindex] = self.pe_array[pe_iindex][pe_jindex].result

                self.utilization_per_pe[pe_iindex][pe_jindex] = self.pe_array[pe_iindex][pe_jindex].mac_utility.count(1) / self.clock

        SystolicArrayLogger.info("Final Clock: {}".format(self.clock))
        SystolicArrayLogger.info("Clock Cycles Per Matrix On Average: {}".format(self.clock / self.thread_count))
        SystolicArrayLogger.info("Utilization Per PE:\n{}".format(self.utilization_per_pe))
        SystolicArrayLogger.info("Average, Std Utilization Per PE For Systolic Array: {}, {}".format(self.utilization_per_pe.mean(), self.utilization_per_pe.std()))
        SystolicArrayLogger.info("\n\nResults:\n{}".format(self.results))


if __name__ == '__main__':
    pass
