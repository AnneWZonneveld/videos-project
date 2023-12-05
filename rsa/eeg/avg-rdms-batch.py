"""
Averages RDMs per participant per distance type over permutations. 

"""
import glob
import numpy as np
import pickle

distance_types = ['classification', 'dv-classification']
n_subs = 3

for sub in range(n_subs):

    sub_format = format(sub + 1, '02')
    

    for distance_type in distance_types:

        print(f'sub {sub_format} {distance_type}')

        file_dir =  f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/{distance_type}/' 
        files = []

        for file in glob.glob(file_dir + '*'): 
            files.append(file)
        
        perm_array = np.zeros((len(files), 1000, 1000, 185))

        for i in range(len(files)):

            file = files[i]
            with open(file, 'rb') as f: 
                data = pickle.load(f)
            
            perm_array[i, :, :, :] = data['rdms_array']
        
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
            'info': data['info']
        }

        with open(f'/scratch/azonneveld/rsa/eeg/rdms/sub-{sub_format}/avg_{distance_type}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)




            



