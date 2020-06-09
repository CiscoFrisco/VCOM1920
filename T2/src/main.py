import keras.backend.tensorflow_backend as tfback
import tensorflow as tf
import sys
import os
from task1 import task1_CNN, task1_BOVW
from task2 import task2
from task3 import task3


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

# Comment lines 24 and 25 to use the CPU OR comment line 26 to use the GPU
tfback._get_available_gpus = _get_available_gpus
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    if (len(sys.argv) != 2 and len(sys.argv) != 3) or (sys.argv[1] != '1' and sys.argv[1] != '2' and sys.argv[1] != '3'):
        print("Usage: python " + sys.argv[0] + " <TASK>")
        print("Where TASK is one of 1, 2 or 3.")
        return

    task = sys.argv[1]

    if task == '1':
        strategy = sys.argv[2]
        if strategy == 'bovw':
            task1_BOVW()
        else:
            task1_CNN()
    elif task == '2':
        task2()
    else:
        task3()


if __name__ == "__main__":
    main()
