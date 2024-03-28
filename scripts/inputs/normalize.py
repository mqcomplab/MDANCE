from mdance.scripts import normalize

# System info - EDIT THESE
data_file = '../../examples/2D/blob_disk.csv'
delimiter=','
output_base_name = 'array'

if __name__ == '__main__':
    normalize(data_file,delimiter,output_base_name)