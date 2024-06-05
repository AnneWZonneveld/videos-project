"""
Creating group level RDMs based on subject-level RDMs.

Parameters
----------

data_split: str
    Train or test. 
distance_type: str
    Whether to base the RDMs on 'euclidean', 'euclidean-cv', 'classification' (a.k.a. decoding accuracy), or 'dv-classification' 
rois: str
    String with names of all ROIs to compute RDMs for.

"""


import glob
import numpy as np
import pickle
import sys
import shutil
import argparse
import time
import os
from scipy.spatial.distance import squareform

n_subs = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--distance_type', default='euclidean-cv', type=str) 
parser.add_argument('--rois', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

if args.data_split == 'train':
      n_conditions = 1000
elif args.data_split == 'test':
      n_conditions = 102

######################### Load files ##################################
rois = args.rois.split(',')     

GA_data = {}
for roi in rois:
    subs_array = np.zeros((n_subs, n_conditions, n_conditions))
    print(f'Loading {roi}')

    for i in range(n_subs):

        sub = i + 1
        sub_format = format(sub, '02')

        try: 
            file = f'/scratch/azonneveld/rsa/fmri/rdms/sub-{sub_format}/{args.distance_type}/{args.data_split}_rdms.pkl'

            with open(file, 'rb') as f: 
                data = pickle.load(f)

            rdm = data['results'][roi]
            subs_array[i, :, :] = rdm  
        except:
             print(f'sub {sub_format} {roi} missing')
        
    avg_array = np.mean(subs_array, axis=0)
    GA_data[roi] = avg_array
    print(f'{roi} done')

############################ Save results ############################
results_dict = {
    'data_split': args.data_split, 
    'distance_type': args.distance_type,
    'rdm_array': GA_data,
}

res_folder = f'/scratch/azonneveld/rsa/fmri/rdms/GA/{args.distance_type}/'
if os.path.isdir(res_folder) == False:
	os.makedirs(res_folder)
res_file = res_folder + f'GA_{args.data_split}_rdm.pkl'

with open(res_file, 'wb') as f:
    pickle.dump(results_dict, f)
        
        # # Delete individual pseudotrial batches
        # shutil.rmtree(file_dir)
