import os

import numpy as np

from mdance.inputs.preprocess import normalize_file
from mdance import data


# System parameters
data_file = data.blob_disk                          # Input data as a CSV file
array = np.genfromtxt(data_file, delimiter=',')
output_base_name = 'array'                          # Prefix for the output files


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    output_name = f'outputs/{output_base_name}.npy'
    normed_data, min, max, avg = normalize_file(array, norm_type='v3')
    np.save(output_name, normed_data)
