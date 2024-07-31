import numpy as np

from mdance.inputs.preprocess import normalize_file
from mdance import data


# System info - EDIT THESE
data_file = data.blob_disk
array = np.genfromtxt(data_file, delimiter=',')
output_base_name = 'array'


if __name__ == '__main__':
    output_name = f'{output_base_name}.npy'
    normed_data, min, max, avg = normalize_file(array, norm_type='v3')
    np.save(output_name, normed_data)