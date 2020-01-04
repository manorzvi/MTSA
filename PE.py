import logging

PELogger = logging.getLogger("PELogger")


class PE:
    """
    PE logic element in SystolicArray
    """

    def __init__(self, i, j, thread_count, matrix_size, log):
        """
        Construct PE instance.
        PE is basically MAC unit that can multiply 2 scalars from it's west and north corners and accumulate the result
        with previous results.
        It save intermediate results in <threads_number> size array.
        If at least one of its west,north entrances equals zero, it can skip it, and perform the next thread data.
        """
        self.iindex = i
        self.jindex = j

        self.west_buffer  = None
        self.north_buffer = None
        self.east_buffer  = None
        self.south_buffer = None

        self.result = [0 for _ in range(thread_count)]
        self.matrix_size = matrix_size

        self.thread_count = thread_count
        self.onThread = 0

        # mac_utility is binary list. Each clock cycle, '1' is append if the MAC was enabled, '0' otherwise.
        self.mac_utility = []

        if log:
            PELogger.info("PE <{},{}> Initialized.".format(self.iindex, self.jindex))

    def connect(self, west_buffer, north_buffer, east_buffer, south_buffer, log):
        """
        Connect PE to adjacent Buffers.
        """
        self.west_buffer  = west_buffer
        self.north_buffer = north_buffer
        self.east_buffer  = east_buffer
        self.south_buffer = south_buffer

        if log:
            PELogger.debug("<{},{}> Connected: west: {}, north: {}, east: {}, south: {}".format(self.iindex,
                                                                                                self.jindex,
                                                                                                self.west_buffer,
                                                                                                self.north_buffer,
                                                                                                self.east_buffer,
                                                                                                self.south_buffer))

    def tock(self, log):
        """
        basically a Round-Robin mechanism to iterate through threads.
        iterate through north and west input buffers threads until both buffers tops aren't zeroed or None.
        For each such couple, it performs a single MAC move, save the results in local result list, and pass the arguments
        to south and west output buffers, accordingly.
        For zeroed / None couples, it just pass the arguments to the next buffers.
        :return: None
        """

        # Reorder buffers list to start at onThread buffers.
        buffer_reorder = list(self.west_buffer.buffer.copy())[self.onThread:]
        buffer_reorder += list(self.west_buffer.buffer.copy())[:self.onThread]

        MAC_on = False # to indicate if a mac op have took place already this CC

        for thread_id in buffer_reorder:

            west_thread_buffer = self.west_buffer.buffer[thread_id]
            north_thread_buffer = self.north_buffer.buffer[thread_id]

            thread_number = int(thread_id[1:])

            # Try to pull input from west buffer
            try:
                west_in = west_thread_buffer.pop(0)
            except IndexError:

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, West Buffer is Empty.".format(self.iindex, self.jindex, thread_number))
                continue
            # Try to pull input from north buffer
            try:
                north_in = north_thread_buffer.pop(0)
            except IndexError:

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, North Buffer is Empty.".format(self.iindex, self.jindex, thread_number))

                # If we got up here, it means that west succeeded. But if north failed, we need to push back west.
                west_thread_buffer.insert(0, west_in)
                if log:
                    PELogger.debug("<{},{}> - Thread: {}, West Input Pushed Back to West Buffer".format(self.iindex, self.jindex, thread_number))
                continue

            if log:
                PELogger.info("<{},{}> - Thread: {}, West Literal: {}, North Literal: {}".format(self.iindex, self.jindex, thread_number, west_in, north_in))

            # If MAC_on=True, that mean that the MAC has already worked this clock cycle.
            # If that's the case, if inputs aren't zeros/bubbles, we need to push them beck to input buffers.
            if west_in != 0 and north_in != 0 and west_in is not None and north_in is not None and MAC_on:

                west_thread_buffer.insert(0, west_in)

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, West Input Pushed Back to West Buffer".format(self.iindex, self.jindex, thread_number))

                north_thread_buffer.insert(0, north_in)

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, North Input Pushed Back to North Buffer".format(self.iindex, self.jindex, thread_number))

                continue

            # If both aren't zero, aren't None (=bubble), and MAC hasn't worked yet this clock cycle, Calculate.
            # Turn MAC_on to True to indicate that MAC is working this clock cycle.
            elif west_in != 0 and north_in != 0 and west_in is not None and north_in is not None and not MAC_on:

                self.onThread += 1
                if self.onThread > self.thread_count - 1:
                    self.onThread = 0

                MAC_on = True

                # Multiply west input by north input. Save intermediate result in local register <thread_number> index.
                self.result[thread_number] += west_in * north_in

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, "
                                   "MAC On, Intermediate result: {}".format(self.iindex, self.jindex, thread_number, self.result[thread_number]))
                # Push west input to east buffer
                self.east_buffer.push_to(thread_number, west_in, log=log)

                # Push north input to south buffer
                self.south_buffer.push_to(thread_number, north_in, log=log)

                # Add '1' to mac_utility to indicate that the MAC worked on this cycle.
                self.mac_utility.append(1)

                continue

            # If one of them or both are zero, result already known.
            # Just push west input to east buffer, north input to south buffer.
            elif west_in == 0 or north_in == 0:

                # Push west input to east buffer
                self.east_buffer.push_to(thread_number, west_in, log=log)

                # Push north input to south buffer
                self.south_buffer.push_to(thread_number, north_in, log=log)

                continue

            # If one of them or both are bubbles, non need to calculate result.
            # Just push west input to east buffer, north input to south buffer.
            elif west_in is None and north_in is None:

                # Push west input to east buffer
                self.east_buffer.push_to(thread_number, west_in, log=log)

                # Push north input to south buffer
                self.south_buffer.push_to(thread_number, north_in, log=log)

                continue

        # If after we passed through all Threads buffers, and yet MAC_on didn't change,
        # that's mean that all inputs were zeros or bubbles.
        # In that case, MAC is not working in this clock cycle, and we append '0' to utility logger.
        if not MAC_on:
            self.mac_utility.append(0)

        if log:
            PELogger.info("<{},{}> - All Threads, Intermediate Result: {}, MAC Utilization History: {}".format(self.iindex, self.jindex, self.result, self.mac_utility))

    def __repr__(self):
        return '\n<{},{}>:\n\tWestBuffer: {}\n\tNorthBuffer: {}\n\t' \
                             'EastBuffer: {}\n\tSouthBuffer: {}\n\t' \
                             'Result: {}'.format(self.iindex,       self.jindex,       self.west_buffer,
                                                 self.north_buffer, self.east_buffer,  self.south_buffer, self.result)


class PElimited(PE):
    """
    Special PE to support BUFFERlimited
    """

    def __init__(self, i, j, thread_count, matrix_size, log):

        super().__init__(i=i, j=j, thread_count=thread_count, matrix_size=matrix_size, log=log)

        if log:
            PELogger.info("PE Changed To PElimited")

    def tock(self, log):
        """
        overrides original tock method to support limited buffers.
        The main difference is the check for output sizes.
        in case that the inputs are good to go, but correspondent output are full, next thread is being checked.
        """

        buffer_reorder = list(self.west_buffer.buffer.copy())[self.onThread:]
        buffer_reorder += list(self.west_buffer.buffer.copy())[:self.onThread]

        MAC_on = False

        for thread_id in buffer_reorder:

            west_thread_buffer = self.west_buffer.buffer[thread_id]
            north_thread_buffer = self.north_buffer.buffer[thread_id]

            thread_number = int(thread_id[1:])

            # Try to pull input from west buffer
            try:
                west_in = west_thread_buffer.pop(0)
            except IndexError:

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, West Buffer is Empty.".format(self.iindex, self.jindex, thread_number))
                continue

            # Try to pull input form north buffer
            try:
                north_in = north_thread_buffer.pop(0)
            except IndexError:

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, North Buffer is Empty.".format(self.iindex, self.jindex, thread_number))
                west_thread_buffer.insert(0, west_in)

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, West Input Pushed Back to West Buffer".format(self.iindex, self.jindex, thread_number))
                continue

            if log:
                PELogger.info("<{},{}> - Thread: {}, West Literal: {}, North Literal: {}".format(self.iindex, self.jindex, thread_number, west_in, north_in))

            # If MAC_on=True, that mean that the MAC has already worked this clock cycle.
            # If that's the case, if inputs aren't zeros/bubbles, we need to push them beck to input buffers.
            if west_in != 0 and north_in != 0 and west_in is not None and north_in is not None and MAC_on:

                west_thread_buffer.insert(0, west_in)

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, {} Pushed Back to West Buffer".format(self.iindex, self.jindex, thread_number, west_in))

                north_thread_buffer.insert(0, north_in)

                if log:
                    PELogger.debug("<{},{}> - Thread: {}, {} Input Pushed Back to North Buffer".format(self.iindex, self.jindex, thread_number, north_in))

                continue

            # If both aren't zero, aren't None (=bubble), and MAC hasn't worked yet this clock cycle, Calculate.
            # Addition for PElimited only: calculate only if there are enough space in the output buffers for the literals. Otherwise, push back to input buffers.
            # Turn MAC_on to True to indicate that MAC is working this clock cycle.
            if west_in != 0 and north_in != 0 and west_in is not None and north_in is not None and not MAC_on:

                # If One of the Output Channels are full, we can't perform computation for this Channel.
                # Those We push back west_in & north_in.
                if self.east_buffer.is_full(threadID=int(thread_id[1:]), log=log) or self.south_buffer.is_full(threadID=int(thread_id[1:]), log=log):

                    west_thread_buffer.insert(0, west_in)

                    if log:
                        PELogger.debug("<{},{}> - Thread: {}, {} Pushed Back to West Buffer".format(self.iindex, self.jindex, thread_number, west_in))

                    north_thread_buffer.insert(0, north_in)

                    if log:
                        PELogger.debug("<{},{}> - Thread: {}, {} Pushed Back to North Buffer".format(self.iindex, self.jindex, thread_number, north_in))

                    continue

                else:

                    self.onThread += 1
                    if self.onThread > self.thread_count - 1:
                        self.onThread = 0

                    MAC_on = True

                    # Multiply west input by north input. Save intermediate result in local register <thread_number> index.
                    self.result[thread_number] += west_in * north_in

                    if log:
                        PELogger.debug("<{},{}> - Thread: {}, "
                                       "MAC On, Intermediate result: {}".format(self.iindex, self.jindex, thread_number, self.result[thread_number]))
                    # Push west input to east buffer
                    self.east_buffer.push_to(thread_number, west_in, log=log)

                    # Push north input to south buffer
                    self.south_buffer.push_to(thread_number, north_in, log=log)

                    # Add '1' to mac_utility to indicate that the MAC worked on this cycle.
                    self.mac_utility.append(1)

                    continue

            # If one of them or both are zero, result already known.
            # PElimited Addition: check if there is enough space in next buffers. if not, push inputs back.
            # Otherwise, Just push west input to east buffer, north input to south buffer.
            elif west_in == 0 or north_in == 0:

                # If One of the Output Channels are full, we can't push even zeros to this Channel.
                # Those We push back west_in & north_in.
                if self.east_buffer.is_full(threadID=int(thread_id[1:]), log=log) or self.south_buffer.is_full(threadID=int(thread_id[1:]), log=log):

                    west_thread_buffer.insert(0, west_in)

                    if log:
                        PELogger.debug("<{},{}> - Thread: {}, {} Pushed Back to West Buffer".format(self.iindex, self.jindex,  thread_number, west_in))

                    north_thread_buffer.insert(0, north_in)

                    if log:
                        PELogger.debug("<{},{}> - Thread: {}, {} Pushed Back to North Buffer".format(self.iindex, self.jindex, thread_number, north_in))

                    continue

                else:

                    # Push west input to east buffer
                    self.east_buffer.push_to(thread_number, west_in, log=log)

                    # Push north input to south buffer
                    self.south_buffer.push_to(thread_number, north_in, log=log)

                    continue

            # If one of them or both are bubbles, non need to calculate result.
            # Just push west input to east buffer, north input to south buffer.
            elif west_in is None and north_in is None:

                # Push west input to east buffer
                self.east_buffer.push_to(thread_number, west_in, log=log)

                # Push north input to south buffer
                self.south_buffer.push_to(thread_number, north_in, log=log)

                continue

        # If after we passed through all Threads buffers, and yet MAC_on didn't change,
        # that's mean that all inputs were zeros or bubbles, or all output buffers are full.
        # In that case, MAC is not working in this clock cycle, and we append '0' to utility logger.
        if not MAC_on:
            self.mac_utility.append(0)

        if log:
            PELogger.info("<{},{}> - All Threads, Intermediate Result: {}".format(self.iindex, self.jindex, self.result))
