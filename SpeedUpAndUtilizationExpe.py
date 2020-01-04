import numpy as np
import os
import json
from pprint import pprint
from MTSA_generator_script import *
import re
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.nan)


def generate_multiple_runs():
    """
    - Configurations according to config dictionary down here.
    - Create work area based on Configurations, and step into it. save configurations in it.
    - For each sparsity and threads number, create sub - work dir, step into it and apply run_once, with those parameters.
    :return:
    """

    configExp = {'array_size'      : 8,
                 'buffer_depth'    : 2,
                 'top_value'       : 10,
                 'sparsity_values' : list(np.linspace(0, 0.96, 24)),
                 'input_times'     : 200,
                 'threads'         : [1, 2, 4, 8, 16]}

    configExp['values'] = np.arange(configExp['top_value'])

    workdir = 'MTSA_{}X{}SA_'.format(configExp['array_size'], configExp['array_size'])
    if configExp['buffer_depth'] == -1:
        workdir += 'BUFFINF_'
    else:
        workdir += 'BUFFLIM{}_'.format(configExp['buffer_depth'])
    workdir += '{}X{}WEST_{}X{}NORTH_'.format(configExp['array_size'],
                                              configExp['array_size'] * configExp['input_times'],
                                              configExp['array_size'] * configExp['input_times'],
                                              configExp['array_size'])
    for t in configExp['threads']:
        workdir += '{}_'.format(t)
    workdir = workdir[:-1]
    workdir += 'THREADS'

    if not os.path.exists(workdir):
        try:
            os.makedirs(workdir)
        except OSError:
            print("[ERROR] - Can't Create " + workdir + " Directory.")
            exit(1)

        os.chdir(workdir)
    else:
        try:
            os.chdir(workdir)
        except OSError:
            print("[ERROR] - Can't Change Directory To " + workdir)
            exit(2)

    np.save('ExpConfigFile', configExp)

    for sparsity in configExp['sparsity_values']:

        for t in configExp['threads']:

            configRun = {'thread_number' : t,
                         'array_size'    : configExp['array_size'],
                         'sparsity'      : sparsity,
                         'buffer_depth'  : configExp['buffer_depth'],
                         'inputMultiplier' : configExp['input_times'],
                         'loggingNow'      : False
                         }
            if configExp['buffer_depth'] < 0:
                configRun['is_limited_buffer'] = 'No'
            else:
                configRun['is_limited_buffer'] = 'Yes'

            rundir = 'MTSA_{}X{}SA_'.format(configRun['array_size'] , configRun['array_size'])
            if configRun['buffer_depth'] == -1:
                rundir += 'BUFFINF_'
            else:
                rundir += 'BUFFLIM{}_'.format(configRun['buffer_depth'])

            rundir += '{}X{}WEST_{}X{}NORTH_'.format(configRun['array_size'] ,
                                                     configRun['array_size'] * configRun['inputMultiplier'] ,
                                                     configRun['array_size'] * configRun['inputMultiplier'] ,
                                                     configRun['array_size'])
            rundir += '{0:.2f}SPARS_'.format(sparsity).replace('.', '_')
            rundir += '{}THREAD'.format(t)

            if not os.path.exists(rundir):
                try:
                    os.makedirs(rundir)
                except OSError:
                    print("[ERROR] - Can't Create " + rundir + " Directory.")
                    exit(3)

                os.chdir(rundir)
            else:
                try:
                    os.chdir(rundir)
                except OSError:
                    print("[ERROR] - Can't Change Directory To " + rundir)
                    exit(4)

            with open('ConfigFile.json', 'w') as js:
                json.dump(configRun, js)

            os.chdir('..')

    dirList = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.' , d))]

    for d in dirList:

        print('RunDir:' + d)

        runOnce(dumpTo=d)

        os.chdir('..')


def plot_speedup_and_util_improvement_graph(workdir):
    """
    - Iterate through all Experiments in work directory, from which extract relevant information (saved in summary file).
    - Gather all information together, and plot:
        - speedUp plot.
        - absolute average clock cycles plot.
        - Utilization plot.
        - Utilization Improvement plot.
    - if more results gathered per experiment, average over them, and then add to plot.

    :param workdir: work directory
    :return:
    """

    try:
        os.chdir(workdir)

    except OSError:
        print("[ERROR] - Can't Change Directory To " + workdir)
        exit(10)

    configExp = dict()
    try:
        configExp = np.load('ExpConfigFile.npy')
    except IOError:
        print("[ERROR] - Configuration Doe's not exist")
        exit(11)

    dirList    = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.' , d))]
    sparsities = []

    avg_clock_per_matrix_per_thread_per_sparsity  = list()
    total_avg_utilization_per_thread_per_sparsity = list()

    for d in dirList:

        sparsity_thread_regex = re.match('^MTSA_.*SA_BUFF.*WEST_.*NORTH_(.*)SPARS_{}THREAD$'.format(sorted(configExp.item().get('threads'))[0]) , d)

        if sparsity_thread_regex:

            sparsities.append(float(sparsity_thread_regex.group(1).replace('_', '.')))

            print('[INFO] - read from: ' + d)
            print('-------------------------------------------------------------------------------')
            try:
                os.chdir(d)

            except OSError:
                print("[ERROR] - Can't Change Directory To " + d)
                exit(12)

            summary_over_thread_per_sparsity_sample(avg_clock_per_matrix_per_thread_per_sparsity=avg_clock_per_matrix_per_thread_per_sparsity,
                                                    total_avg_utilization_per_thread_per_sparsity=total_avg_utilization_per_thread_per_sparsity)

            os.chdir('..')

    avg_clock_per_matrix_per_thread_per_sparsity_for_plot  = np.asarray(avg_clock_per_matrix_per_thread_per_sparsity)
    avg_clock_per_matrix_per_thread_per_sparsity_for_plot = \
        avg_clock_per_matrix_per_thread_per_sparsity_for_plot.reshape(1, len(avg_clock_per_matrix_per_thread_per_sparsity_for_plot))

    total_avg_utilization_per_thread_per_sparsity_for_plot = np.asarray(total_avg_utilization_per_thread_per_sparsity)
    total_avg_utilization_per_thread_per_sparsity_for_plot = \
        total_avg_utilization_per_thread_per_sparsity_for_plot.reshape(1, len(total_avg_utilization_per_thread_per_sparsity_for_plot))

    for t in sorted(configExp.item().get('threads'))[1:]:

        avg_clock_per_matrix_per_thread_per_sparsity = list()
        total_avg_utilization_per_thread_per_sparsity = list()

        for d in dirList:

            sparsity_thread_regex = re.match('^MTSA_(.*)SA_BUFF.*WEST_(.*)NORTH_(.*)SPARS_{}THREAD$'.format(t), d)

            if not sparsity_thread_regex:
                continue

            elif sparsity_thread_regex:
                print('[INFO] - read from: ' + d)
                print('-------------------------------------------------------------------------------')
                try:
                    os.chdir(d)

                except OSError:
                    print("[ERROR] - Can't Change Directory To " + d)
                    exit(12)

            summary_over_thread_per_sparsity_sample(avg_clock_per_matrix_per_thread_per_sparsity=avg_clock_per_matrix_per_thread_per_sparsity ,
                                                    total_avg_utilization_per_thread_per_sparsity=total_avg_utilization_per_thread_per_sparsity)
            os.chdir('..')

        v1 = np.asarray(avg_clock_per_matrix_per_thread_per_sparsity)
        v2 = np.asarray(total_avg_utilization_per_thread_per_sparsity)

        v1 = v1.reshape(1, len(v1))
        v2 = v2.reshape(1, len(v2))

        avg_clock_per_matrix_per_thread_per_sparsity_for_plot  = np.concatenate((avg_clock_per_matrix_per_thread_per_sparsity_for_plot,  v1), axis=0)
        total_avg_utilization_per_thread_per_sparsity_for_plot = np.concatenate((total_avg_utilization_per_thread_per_sparsity_for_plot, v2), axis=0)

        del v1, v2

    plot_speedup(Y=avg_clock_per_matrix_per_thread_per_sparsity_for_plot,  x=sparsities,  mode='speedup',                 threads=sorted(configExp.item().get('threads')), mode2='sparsity')
    plot_speedup(Y=avg_clock_per_matrix_per_thread_per_sparsity_for_plot,  x=sparsities,  mode='clock',                   threads=sorted(configExp.item().get('threads')), mode2='sparsity')
    plot_speedup(Y=total_avg_utilization_per_thread_per_sparsity_for_plot, x=sparsities,  mode='utilization_improvement', threads=sorted(configExp.item().get('threads')), mode2='sparsity')
    plot_speedup(Y=total_avg_utilization_per_thread_per_sparsity_for_plot, x=sparsities,  mode='utilization',             threads=sorted(configExp.item().get('threads')), mode2='sparsity')


def summary_over_thread_per_sparsity_sample(avg_clock_per_matrix_per_thread_per_sparsity, total_avg_utilization_per_thread_per_sparsity):

    summaryList = [f for f in os.listdir('.') if (os.path.isfile(os.path.join('.' , f)) and re.match('^Summary.*.npy$' , f))]

    tmp_avg_clock_per_matrix = 0
    tmp_total_avg_utilization = 0

    for f in summaryList:
        summary = np.load(f)

        tmp_avg_clock_per_matrix += summary.item().get('avg_clock_per_matrix')
        tmp_total_avg_utilization += summary.item().get('total_avg_utilization')

    tmp_avg_clock_per_matrix /= len(summaryList)
    tmp_total_avg_utilization /= len(summaryList)

    avg_clock_per_matrix_per_thread_per_sparsity.append(tmp_avg_clock_per_matrix)
    total_avg_utilization_per_thread_per_sparsity.append(tmp_total_avg_utilization)

    return avg_clock_per_matrix_per_thread_per_sparsity, total_avg_utilization_per_thread_per_sparsity


def plot_speedup(Y, x, mode, threads, mode2):

    f, ax = plt.subplots(figsize=(16, 16))

    title = '_'.join(os.getcwd().split('\\')[-1].split('_')[:5])

    if mode == 'speedup':
        normalized_Y = 1 / np.divide(Y, Y[0])
        ax.set_ylabel('Speedup Over 1 Thread')
        title += '_SPEEDUP'
        ax.set_title(title)

        data2text = np.asarray(x).reshape(1, len(x))
        data2text = np.concatenate((data2text, normalized_Y), axis=0)
        np.savetxt('data_speedup', data2text)

    elif mode == 'utilization_improvement':
        normalized_Y = np.divide(Y, Y[0])
        ax.set_ylabel('Utilization Improvement over 1 Thread')
        title += '_UTIL_IMPROVE'
        ax.set_title(title)

        data2text = np.asarray(x).reshape(1 , len(x))
        data2text = np.concatenate((data2text , normalized_Y) , axis=0)
        np.savetxt('data_utilization_improvement' , data2text)

    elif mode == 'clock':
        normalized_Y = Y
        ax.set_ylabel('Clocks')
        title += '_ABS_CLOCKS'
        ax.set_title(title)

        data2text = np.asarray(x).reshape(1 , len(x))
        data2text = np.concatenate((data2text , normalized_Y) , axis=0)
        np.savetxt('data_clock' , data2text)

    elif mode == 'utilization':
        normalized_Y = Y
        ax.set_ylabel('Utilization')
        title += '_ABS_UTIL'
        ax.set_title(title)

        data2text = np.asarray(x).reshape(1 , len(x))
        data2text = np.concatenate((data2text , normalized_Y) , axis=0)
        np.savetxt('data_utilization' , data2text)
    else:
        normalized_Y = Y

    for idx, t in zip(range(len(normalized_Y)), threads):
        y = normalized_Y[idx]
        ax.plot(x, y, label='{} Threads'.format(t))
        if mode2 == 'sparsity':
            if idx == 0:
                ax.plot(x[0], y[0], '.', c='black')
                ax.text(x[0], y[0], '{0:.2f},{1:.2f}'.format(x[0], y[0]))

            ax.plot(x[-1] , y[-1] , '.' , c='black')
            ax.text(x[-1] , y[-1] , '{0:.2f},{1:.2f}'.format(x[-1] , y[-1]))

        elif mode2 == 'buffer_lim':
            for i in range(len(x)):
                ax.plot(x[i] , y[i], '.', c='black')
                ax.text(x[i] , y[i] , '{0:.2f},{1:.2f}'.format(x[i] , y[i]))

    if mode2 == 'sparsity':
        ax.set_xlabel('Sparsity\n(as Probability for zero)')
    elif mode2 == 'buffer_lim':
        ax.set_xlabel('Buffer Maximal Length')

    ax.legend()

    plt.savefig(title)

    plt.show()


if __name__ == '__main__':
    generate_multiple_runs()
    #plot_speedup_and_util_improvement_graph(workdir='MTSA_8X8SA_BUFFLIM2_8X1600WEST_1600X8NORTH_1_2_4_8_16THREADS')

