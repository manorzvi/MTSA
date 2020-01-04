import os
import json
import numpy as np
from SystolicArray import SystolicArray
from pprint import pprint
from datetime import datetime


def runOnce(dumpTo='Default_WorkArea'):
    """
    - Create work dir 'dumpTo' if not exist, step into it.
    - If exist, check if exist config file. if so: read it, else: prompt the user to specify properties for the simulator.
    - Generate single experiment according to those parameters.
    - Save summary into work area directory.
    :param dumpTo: Location directory to save results into
    """

    if not os.path.exists(dumpTo):
        try:
            os.makedirs(dumpTo)
        except OSError:
            print("[ERROR] - Can't Create " + dumpTo + " Directory.")
            exit(1)

        os.chdir(dumpTo)
    else:
        try:
            os.chdir(dumpTo)
        except OSError:
            print("[ERROR] - Can't Change Directory To " + dumpTo)
            exit(2)

        # Check if ConfigFile exist in directory
        fileList = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
        configDict = dict()
        found = False
        for f in fileList:
            if f.endswith('.json'):
                print('[INFO] - Configuration File Found.')
                with open(f, 'r') as js:
                    try:
                        configDict = json.load(js)
                    except Exception:
                        print('JSON Configuration file Corrupted or something')
                        break
                print('[INFO] - Configurations:\n------------------------')
                pprint(configDict)
                found = True
                break

        if not found:
            print("[INFO] - Configuration file didn't found")
            configDict['thread_number']     = int(input('How Many Threads?'))
            configDict['array_size']        = int(input('What is the size of the Systolic Array? Enter 1 number - array must be square'))
            configDict['sparsity']          = float(input('What is the sparsity level? Enter number in range [0,1] to indicate the probability for zero'))
            configDict['is_limited_buffer'] = input('Are the buffers depth limited? (Yes/No)')
            if configDict['is_limited_buffer'] == 'Yes':
                configDict['is_limited_buffer'] = True
            elif configDict['is_limited_buffer'] == 'No':
                configDict['is_limited_buffer'] = False
            if configDict['is_limited_buffer']:
                configDict['buffer_depth']  = int(input('What is the limit? Enter some Integer in range [2, inf].\n'))
            else:
                configDict['buffer_depth'] = -1
            configDict['inputMultiplier'] = int(input('How long are the inputs? Enter an integer to multiply SA edge by.\n'
                                                      'For example: for 8X8 SA, 8X800 west input tensors and 800X8 north input tensors, Enter 100.'))
            configDict['loggingNow'] = input("Want's to log progress (Yes/No)? Note that it might make the simulation approx 16 time slower")
            if configDict['loggingNow'] == 'Yes':
                configDict['loggingNow'] = True
            elif configDict['loggingNow'] == 'No':
                configDict['loggingNow'] = False

            with open('ConfigFile.json', 'w') as js:
                json.dump(configDict, js)

        top_value = 10
        values    = np.arange(top_value)  # Matrices values would rand from: [0,top_value]

        # We consider specified probability for 'zero' and Uniform Distribution on the rest.
        probabilities = [configDict['sparsity']] + [(1 - configDict['sparsity']) / values[1:].shape[0] for _ in values[1:]]
        print('Over Distribution: {}'.format(['{0:.2}'.format(p) for p in probabilities]))

        west_tensor_shape  = (configDict['thread_number'], configDict['array_size'],                                 configDict['array_size'] * configDict['inputMultiplier'])
        north_tensor_shape = (configDict['thread_number'], configDict['array_size'] * configDict['inputMultiplier'], configDict['array_size'])

        data_matrices   = np.random.choice(values, west_tensor_shape,  p=probabilities)
        weight_matrices = np.random.choice(values, north_tensor_shape, p=probabilities)
        result_matrices = np.matmul(data_matrices, weight_matrices)

        systolic_array = SystolicArray(west_matrices=data_matrices,
                                       north_matrices=weight_matrices,
                                       array_size=configDict['array_size'],
                                       thread_count=configDict['thread_number'],
                                       buffer_depth=configDict['buffer_depth'],
                                       log=configDict['loggingNow'])

        while 1:

            systolic_array.tick(log=configDict['loggingNow'])

            if systolic_array.isDone():
                systolic_array.summarize()
                break

        if np.any(systolic_array.results - result_matrices):
            print('[ERROR] - MTSA results are different then Expected results')
            exit(3)

        summaryDict = dict()

        summaryDict['total_clock'] = systolic_array.clock
        summaryDict['avg_clock_per_matrix'] = summaryDict['total_clock'] / configDict['thread_number']
        summaryDict['utilization_per_pe'] = systolic_array.utilization_per_pe
        summaryDict['total_avg_utilization'] = summaryDict['utilization_per_pe'].mean()
        summaryDict['total_std_utilization'] = summaryDict['utilization_per_pe'].std()

        summaryDict['load_record_per_buffer'] = dict()
        for i in range(systolic_array.array_size-1):
            for j in range(systolic_array.array_size-1):
                summaryDict['load_record_per_buffer'][(systolic_array.horizontal_buffer_array[i][j].iindex,
                                                       systolic_array.horizontal_buffer_array[i][j].jindex, 'H')] = systolic_array.horizontal_buffer_array[i][j].load
                summaryDict['load_record_per_buffer'][(systolic_array.vertical_buffer_array[i][j].iindex,
                                                       systolic_array.vertical_buffer_array[i][j].jindex, 'V')] = systolic_array.vertical_buffer_array[i][j].load

        np.save('Summary' + str(datetime.now().month) + '_' + str(datetime.now().day) + '_' + str(datetime.now().year) + '_' +
                str(datetime.now().hour) + '_' + str(datetime.now().minute) + '_' + str(datetime.now().second), summaryDict)


if __name__ == '__main__':
    runOnce()
