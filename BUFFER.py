import logging

BufferLogger = logging.getLogger('BufferLogger')


class BUFFER:
    """
    Data Structure to hold intermediate literals between PE's.
    Has a dictionary, each key is a thread id, each value is a list behave like a Fifo.
    """
    def __init__(self, thread_count, iindex, jindex, log):

        self.buffer = dict()

        # For each thread, count how many data items in buffer, not including None's (None=bubble)
        self.load = dict()

        self.iindex = iindex
        self.jindex = jindex

        for t in range(thread_count):
            self.buffer['T{}'.format(t)] = [None]
            self.load['T{}'.format(t)] = []

        self.depth_limit = None

        if log:
            BufferLogger.debug("BUFFER <{},{}>: {}".format(self.iindex, self.jindex, self.buffer))

    def push_to(self, threadID, value, log):
        """
        try to push value to channel threadID, if log=True, then log a proper message.
        """
        try:
            self.buffer['T{}'.format(threadID)].append(value)

            if log:
                BufferLogger.debug("<{},{}>: Value {} Pushed to Thread: {}".format(self.iindex, self.jindex, value, threadID))

        except KeyError:

            BufferLogger.error('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

            raise ValueError('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

    def __repr__(self):
        return "<{},{}>".format(self.iindex, self.jindex)

    def update_load(self):
        """
        Update load logger according to current buffer state
        """
        for key, buffer in self.buffer.items():
            self.load[key].append(len([1 for i in buffer if i is not None]))


class BUFFERlimited(BUFFER):
    """
    Buffer with limited depth
    """
    def __init__(self, thread_count, depth, iindex, jindex, log):

        super().__init__(thread_count=thread_count, iindex=iindex, jindex=jindex, log=log)

        self.depth_limit = depth

        if log:
            BufferLogger.debug("BUFFER Changed To BUFFERlimited With Limit {}".format(self.depth_limit))

    def push_to(self, threadID, value, log):

        try:

            if len(self.buffer['T{}'.format(threadID)]) < self.depth_limit:

                self.buffer['T{}'.format(threadID)].append(value)

                if log:
                    BufferLogger.info("<{},{}>: Value {} Pushed to Buffer-Thread: {}. "
                                      "Buffer Size Now: {}".format(self.iindex, self.jindex, value, threadID, len(self.buffer['T{}'.format(threadID)])))
                return True

            else:
                if log:
                    BufferLogger.info("BUFFERlimited <{},{}>: Value {} Didn't push to Thread: {} because it's full".format(self.iindex, self.jindex, value, threadID))
                return False

        except KeyError:
            BufferLogger.error('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))
            raise ValueError('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

    def delete_last(self, threadID, log):

        try:

            del self.buffer['T{}'.format(threadID)][-1]

            BufferLogger.info('BUFFERlimited <{},{}>: Last Element Removed From Thread {}'.format(self.iindex, self.jindex, threadID))

        except KeyError:

            BufferLogger.error('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))
            raise ValueError('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

    def is_full(self, threadID, log):

        if len(self.buffer['T{}'.format(threadID)]) < self.depth_limit:

            if log:
                BufferLogger.debug('BUFFERlimited <{},{}> - Thread {} Is Not Full'.format(self.iindex, self.jindex, threadID))
            return False

        else:

            if log:
                BufferLogger.debug('BUFFERlimited <{},{}> - Thread {} Is Full'.format(self.iindex, self.jindex, threadID))
            return True


class FIFO(BUFFER):
    """
    Input FIFO for SystolicArray edges.
    Inherit from BUFFER class, in order to simplify systolicArray build in SystolicArray.py.
    """
    def __init__(self, threads, i, j, thread_count, log):
        """
        Override PE constructor. FIFO constructor called separated from pe_array build,
        and its indexes are fixed to -1.
        """
        super().__init__(thread_count=thread_count, iindex=i, jindex=j, log=log)
        del self.load

        for t, thread in zip(range(len(threads)), threads):
            self.buffer['T{}'.format(t)] = thread

        if log:
            BufferLogger.debug("BUFFER Changed To FIFO: <{},{}>: {}".format(self.iindex, self.jindex, self.buffer))


class OUTPUT(BUFFER):
    """
    Special BUFFER class without None
    """
    def __init__(self, thread_count, iindex, jindex, log):

        super().__init__(thread_count=thread_count, iindex=iindex, jindex=jindex, log=log)

        del self.buffer
        del self.load

        self.buffer = dict()

        for t in range(thread_count):
            self.buffer['T{}'.format(t)] = []

        if log:
            BufferLogger.debug("BUFFER Changed To OUTPUT <{},{}>: {}".format(self.iindex, self.jindex, self.buffer))

    def push_to(self, threadID, value, log):
        try:

            if value is not None:
                self.buffer['T{}'.format(threadID)].append(value)

        except KeyError:

            BufferLogger.error('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

            raise ValueError('Invalid Thread ID ({}) in Buffer <{},{}>'.format(threadID, self.iindex, self.jindex))

    def is_full(self, threadID, log):
        """
        don't care for output buffers fullness.
        """
        return False
