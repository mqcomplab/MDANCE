import numpy as np
import sys
sys.path.insert(0, '../../')
from src.inputs.preprocess import normalize_file

# System info - EDIT THESE
data_file = '../../examples/2D/blob_disk.csv'
array = np.genfromtxt(data_file, delimiter=',')
output_base_name = 'array'

if __name__ == '__main__':
    output_name = f'{output_base_name}.npy'
    normed_data, min, max, avg = normalize_file(array, norm_type='v3')
    np.save(output_name, normed_data)