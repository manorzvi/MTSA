import re
import os
import shutil
from pprint import pprint
import numpy as np
from SpeedUpAndUtilizationExpe import plot_speedup


def speedUp_bufferLimit(workdir=None):

    if not workdir:
        print('[ERROR] - Provide Anchor Directory')
        exit(20)

    workdir_regex = re.match('MTSA_(.*)SA_BUFFALL_(.*)WEST_(.*)NORTH_\d+_\d+SPARS(.*)THREADS', workdir)
    if not workdir_regex:
        print("[ERROR] - Bad Anchor")
        exit(21)

    if not os.path.exists(workdir):
        try:
            os.makedirs(workdir)
        except OSError:
            print("[ERROR] - Can't Create " + workdir + " Directory.")
            exit(22)

    os.chdir(workdir)

    dirList       = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.' , d))]
    buffer_limits = []
    threads       = [1, 2, 4, 8, 16]
    avg_clock_per_matrix_per_thread_per_buffer_limit  = list()
    total_avg_utilization_per_thread_per_buffer_limit = list()

    for d in dirList:

        buffer_limits_thread_regex = re.match('^MTSA_.*SA_BUFFLIM(\d+)_.*WEST_.*NORTH_.*SPARS_{}THREAD$'.format(threads[0]) , d)

        if buffer_limits_thread_regex:

            if buffer_limits_thread_regex.group(1) == 'INF':
                print('[EROOR] - OopsiPoopsi')
                exit(23)
            elif buffer_limits_thread_regex.group(1).isdigit():
                buffer_limits.append(int(buffer_limits_thread_regex.group(1)))

            print('[INFO] - read from: ' + d)
            print('-------------------------------------------------------------------------------')
            try:
                os.chdir(d)

            except OSError:
                print("[ERROR] - Can't Change Directory To " + d)
                exit(24)

            summary_over_thread_per_buffer_lim_sample(avg_clock_per_matrix_per_thread_per_buffer_limit=avg_clock_per_matrix_per_thread_per_buffer_limit,
                                                      total_avg_utilization_per_thread_per_buffer_limit=total_avg_utilization_per_thread_per_buffer_limit)

            os.chdir('..')


    avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot = np.asarray(avg_clock_per_matrix_per_thread_per_buffer_limit)
    avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot = \
        avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot.reshape(1 , len(avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot))

    total_avg_utilization_per_thread_per_buffer_limit_for_plot = np.asarray(total_avg_utilization_per_thread_per_buffer_limit)
    total_avg_utilization_per_thread_per_buffer_limit_for_plot = \
        total_avg_utilization_per_thread_per_buffer_limit_for_plot.reshape(1 , len(total_avg_utilization_per_thread_per_buffer_limit_for_plot))

    print(avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot)
    print(total_avg_utilization_per_thread_per_buffer_limit_for_plot)
    print('********')

    for t in threads[1:]:

        avg_clock_per_matrix_per_thread_per_buffer_limit = list()
        total_avg_utilization_per_thread_per_buffer_limit = list()

        for d in dirList:

            buffer_limits_thread_regex = re.match('^MTSA_.*SA_BUFFLIM(\d+)_.*WEST_.*NORTH_.*SPARS_{}THREAD$'.format(t) , d)

            if not buffer_limits_thread_regex:
                continue

            elif buffer_limits_thread_regex:
                print('[INFO] - read from: ' + d)
                print('-------------------------------------------------------------------------------')
                try:
                    os.chdir(d)

                except OSError:
                    print("[ERROR] - Can't Change Directory To " + d)
                    exit(25)

            summary_over_thread_per_buffer_lim_sample(avg_clock_per_matrix_per_thread_per_buffer_limit=avg_clock_per_matrix_per_thread_per_buffer_limit ,
                                                      total_avg_utilization_per_thread_per_buffer_limit=total_avg_utilization_per_thread_per_buffer_limit)
            os.chdir('..')

        v1 = np.asarray(avg_clock_per_matrix_per_thread_per_buffer_limit)
        v2 = np.asarray(total_avg_utilization_per_thread_per_buffer_limit)

        v1 = v1.reshape(1, len(v1))
        v2 = v2.reshape(1, len(v2))

        avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot  = np.concatenate((avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot,  v1), axis=0)
        total_avg_utilization_per_thread_per_buffer_limit_for_plot = np.concatenate((total_avg_utilization_per_thread_per_buffer_limit_for_plot, v2), axis=0)

        del v1, v2

    plot_speedup(Y=avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot,  x=buffer_limits , mode='speedup' ,                 threads=threads, mode2='buffer_lim')
    plot_speedup(Y=avg_clock_per_matrix_per_thread_per_buffer_limit_for_plot,  x=buffer_limits , mode='clock' ,                   threads=threads, mode2='buffer_lim')
    plot_speedup(Y=total_avg_utilization_per_thread_per_buffer_limit_for_plot, x=buffer_limits , mode='utilization_improvement' , threads=threads, mode2='buffer_lim')
    plot_speedup(Y=total_avg_utilization_per_thread_per_buffer_limit_for_plot, x=buffer_limits , mode='utilization' ,             threads=threads, mode2='buffer_lim')


def summary_over_thread_per_buffer_lim_sample(avg_clock_per_matrix_per_thread_per_buffer_limit, total_avg_utilization_per_thread_per_buffer_limit):

    summaryList = [f for f in os.listdir('.') if (os.path.isfile(os.path.join('.' , f)) and re.match('^Summary.*.npy$' , f))]

    tmp_avg_clock_per_matrix = 0
    tmp_total_avg_utilization = 0

    for f in summaryList:
        summary = np.load(f)

        tmp_avg_clock_per_matrix += summary.item().get('avg_clock_per_matrix')
        tmp_total_avg_utilization += summary.item().get('total_avg_utilization')

    tmp_avg_clock_per_matrix /= len(summaryList)
    tmp_total_avg_utilization /= len(summaryList)

    avg_clock_per_matrix_per_thread_per_buffer_limit.append(tmp_avg_clock_per_matrix)
    total_avg_utilization_per_thread_per_buffer_limit.append(tmp_total_avg_utilization)

    return avg_clock_per_matrix_per_thread_per_buffer_limit , total_avg_utilization_per_thread_per_buffer_limit







if __name__ == '__main__':
    speedUp_bufferLimit(workdir='MTSA_8X8SA_BUFFALL_8X1600WEST_1600X8NORTH_0_79SPARS_1_2_4_8_16THREADS')
