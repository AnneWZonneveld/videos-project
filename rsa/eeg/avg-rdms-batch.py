"""
Averages RDMs per participant per distance type over permutations. 

"""
import glob
import numpy as np
import pickle
import sys
import shutil
import argparse

times = 185
n_conditions = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--sfreq', default=500, type=int)
args = parser.parse_args()

sub_format = format(args.sub, '02')
sfreq_format = format(args.sfreq, '04')

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

file_dir =  f'/scratch/azonneveld/rsa/eeg/rdms/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq-{sfreq_format}/'
files = []

for file in glob.glob(file_dir + '*'): 
    files.append(file)

if len(files) != 100:
    print('Found less than 100 files; stop calculation')
    sys.exit()

# Process in batches
perm_array = np.zeros((10, n_conditions, n_conditions, times))

for b in range(10):
    
    b_array = np.zeros((10, n_conditions, n_conditions, times))

    for i in range(10):

        index = (b * 10) + i

        file = files[index]
        with open(file, 'rb') as f: 
            data = pickle.load(f)
        
        for t in range(times):
            b_array[i, :, :, t] = data['rdms_array'][t]
    
    b_avg_array = np.mean(b_array, axis=0)
    perm_array[b, :, :] = b_avg_array

avg_array = np.mean(perm_array, axis=0)

# Save results
results_dict = {
    'data_split': data['data_split'], 
    'sub': data['sub'], 
    'zscore': data['zscore'],
    'distance_type': data['distance_type'],
    'rdms_array': avg_array,
    'times': data['times'],
    'ch_names': data['ch_names'],
    'info': data['info'],
}

with open(f'/scratch/azonneveld/rsa/eeg/rdms/z_{zscore}/sub-{sub_format}/avg_{distance_type}_{sfreq_format}.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
        
        # # Delete individual pseudotrial batches
        # shutil.rmtree(file_dir)




            



