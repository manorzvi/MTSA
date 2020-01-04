from BUFFER import *
import numpy as np


def pack_FIFOs(tensor, axis, thread_count, log):
    """
    Slice 3D ndarray matrix into <array_size>    list of
                                 <threads_count> lists of
                                 <matrix_size>   lists
    w.r.t axis direction.
    For example: if west_matrices = [[[11, 12, 13],
                                      [14, 15, 16],
                                      [17, 18, 19]],
                                     [[21, 22, 23],
                                      [24, 25, 26],
                                      [27, 28, 29]]]
    then west_inputs[0] = [[11, 12, 13],
                           [21, 22, 23]] and
         west_inputs[1] = [[14, 15, 16],
                           [24, 25, 26]] and
         west_inputs[2] = [[17, 18, 19],
                           [27, 28, 29]]
    """
    readyFIFO = []

    if axis == 0:    # west matrix
        for i in range(tensor.shape[1]):
            tmp = tensor[:, i, :]
            tmplist = []

            for j in range(tmp.shape[0]):
                tmplist.append(list(tmp[j, :]))

            for t in range(len(tmplist)):
                for ii in range(i):
                    tmplist[t].insert(0, None)

            readyFIFO.append(FIFO(threads=tmplist, i=i, j=-1, thread_count=thread_count, log=log))

    if axis == 1:    # north matrix
        for i in range(tensor.shape[2]):
            tmp = tensor[:, :, i]
            tmplist = []

            for j in range(tmp.shape[0]):
                tmplist.append(list(tmp[j, :]))

            for t in range(len(tmplist)):
                for ii in range(i):
                    tmplist[t].insert(0, None)

            readyFIFO.append(FIFO(threads=tmplist, i=-1, j=i, thread_count=thread_count, log=log))

    return readyFIFO


def unpack_BUFFERs(buffer_list, axis, matrix_shape):
    """
    unpack fifo dictionary back to numpy.ndarray.
    """
    if axis == 0:   # south fifo_list
        south_output_matrices = np.zeros(matrix_shape)
        for buffer in buffer_list:
            for thread_id, thread_list in buffer.buffer.items():
                if len(thread_list) == south_output_matrices.shape[1]:
                    south_output_matrices[int(thread_id[1:]), :, buffer.jindex] = thread_list
        return south_output_matrices
    if axis == 1:   # east fifo list
        east_output_matrices = np.zeros(matrix_shape)
        for buffer in buffer_list:
            for thread_id, thread_list in buffer.buffer.items():
                if len(thread_list) == east_output_matrices.shape[2]:
                    east_output_matrices[int(thread_id[1:]), buffer.iindex, :] = thread_list
        return east_output_matrices


