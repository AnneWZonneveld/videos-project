""" Averages RDMs per participant per distance type over permutations (that is over different cv-folds)
    Also has to be performed when no cv-measurement is used to tranform data to according format.

	Parameters
	----------
    sub: int
        Subject number
    data_split: str
        Train or test set
    zscore: int
        Z scoring of data (y/n)
    distance_type: str
        Distance type: euclidean, pearson, euclidean-cv, classification or dv-classification
    sfreq: int
        Sampling freqency 
    resampled: int 
        Using resampled frequency RDM (y/n)
    zoom: int
        Using zoomed in RDM (y/n), a.k.a. based on higher sampling frequency for only first second

"""

import os
import glob
import numpy as np
import pickle
import sys
import shutil
import argparse
import time
from scipy.spatial.distance import squareform

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--zscore', default=0, type=int)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--resampled', default=0, type=int)
parser.add_argument('--zoom', default=0, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

sub_format = format(args.sub, '02')
sfreq_format = format(args.sfreq, '04')

if args.zoom == True:
    times = 301
else:
    times = 185
    
if args.data_split == 'train':
    n_conditions = 1000
elif args.data_split == 'test':
    n_conditions = 102

######################### Load files ##################################
if args.resampled == True:
    file_dir =  f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq-{sfreq_format}/resampled/'
elif args.zoom == True:
    file_dir =  f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq-{sfreq_format}/zoom/'
else:
    file_dir =  f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/{args.distance_type}/sfreq-{sfreq_format}/'

files = []
for file in glob.glob(file_dir + '*'): 
    files.append(file)
files.sort()   
if len(files) == 0:
    print(f"No file found for: {file_dir}")

# if args.data_split == 'train':

#     if len(files) != 100:
#         print('Found less than 100 files; stop calculation')
#         print(f'Found {len(files)} files')

#         file_ns = []
#         expected_ns = np.arange(0, 101)
#         for file in files:
#             file_n = int(file.split('/')[-1].split('_')[1])
#             file_ns.append(file_n)

#         missing = list(set(expected_ns).difference(file_ns))
#         print(f'missing files: {missing}')

#         sys.exit()

#     # Process in batches
#     perm_array = np.zeros((10, n_conditions, n_conditions, times))

#     for b in range(10):
#         print(f'processing batch {b}')
        
#         b_array = np.zeros((10, n_conditions, n_conditions, times))

#         for i in range(10):

#             index = (b * 10) + i
#             print(f'index: {index}')

#             file = files[index]
#             with open(file, 'rb') as f: 
#                 data = pickle.load(f)
            
#             for t in range(times):
#                 b_array[i, :, :, t] = data['rdms_array'][t]
        
#         b_avg_array = np.mean(b_array, axis=0)
#         perm_array[b, :, :,  :] = b_avg_array

#     del b_avg_array, b_array

# elif args.data_split == 'test':

perm_array = np.zeros((len(files), n_conditions, n_conditions, times))

for i in range(len(files)):

    file = files[i]
    with open(file, 'rb') as f: 
        data = pickle.load(f)
    
    for t in range(times):
        perm_array[i, :, :, t] = data['rdms_array'][t]

avg_array = np.mean(perm_array, axis=0)
del perm_array
     
# Save results
results_dict = {
    'data_split': 'train', 
    'sub': args.sub, 
    'zscore': args.zscore,
    'distance_type': args.distance_type,
    'rdms_array': avg_array,
    'times': data['times'],
    'ch_names': data['ch_names'],
    'info': data['info'],
}
 
res_folder =  f'/scratch/azonneveld/rsa/eeg/rdms/{args.data_split}/z_{args.zscore}/sub-{sub_format}/'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)


if args.resampled == True:
    file = res_folder + f'avg_{args.distance_type}_{sfreq_format}_res.pkl'
elif args.zoom == True:
    file = res_folder + f'avg_{args.distance_type}_{sfreq_format}_zoom.pkl'
else:
    file = res_folder + f'avg_{args.distance_type}_{sfreq_format}.pkl'

with open(file, 'wb') as f:
    pickle.dump(results_dict, f)
        
        # # Delete individual pseudotrial batches
        # shutil.rmtree(file_dir)




            



