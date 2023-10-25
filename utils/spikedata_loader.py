
""" Load the events data from specified the dataset"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
import re
import random

class DataLoader():
    def __init__(self, config):
        self.data_loc = config["data_loc"]      # Path of the real-valued time-series data
        self.spike_path = config["spike_path"]  # Path where the spike encoded data is stored

        # NOTE: Use the following for DVS style encoding of time-series samples
        # Analog to pulse encoding
        self.sampling_scale = config["timescale"]
        self.vdiff = config["vdiff"]
        self.noise_thresh = config["noise_thresh"]


    def spike_scale(self) -> int:
        """
        Returns the timescale of the spike encoding process.
        """
        return self.sampling_scale

    def encoding_params(self) -> int:
        """
        Returns the spike encoding parameters.
        """
        return self.charge_thresh, self.vdiff, self.sampling_scale, self.spike_path, self.data_loc


    def analog_to_spikes(self, check_cached_data, file_string='recon3D*'): 
        """
        check_cached_data: Check the directory if the spike encoding already exists during testing
        file_string: search for the data files with similar names as in the string key.
        """

        self.spike_data_path = self.spike_path+"_v_"+str(int(self.vdiff))+"_t_"+str(int(self.noise_thresh))+"_scale_"+str(self.sampling_scale)  
        #print('Spike data path:',self.spike_data_path)

        filelist = os.listdir(self.data_loc)
        print('Total files:',len(filelist))
        #print(os.getcwd())
        orig_path= os.getcwd()
        os.chdir(self.data_loc)
        #print(os.getcwd())
        indices = []
        for file1 in glob.glob(file_string):
            path_split = file1.split('_')[1]
            path_split = re.findall(r'\d+',path_split)
            indices.append(path_split[0])
        indices = np.sort(indices)
        os.chdir(orig_path)
        #print('Set the python working directory to default', os.getcwd())

        # Return if spike data is already generated.
        if os.path.exists(self.spike_data_path):
            print('Spike encoded data exists')
            return self.spike_data_path, indices
        
        # Do not generate spike data, but raise an error if data is not found during test.
        if check_cached_data:
            raise ImportError('Data was not created during training, network cannot be tested.')
                
        print("Converting the data into spike with resolution {}".format(self.sampling_scale))
        os.mkdir(self.spike_data_path)

        for k in indices:
            if path_split[-1]=='.csv':
                data_file = self.data_loc + 'recon3D_d'+str(k)+'.csv'
                labels_file = self.data_loc + 'labels_d'+str(k)+'.csv'
                data_in = pd.read_csv(data_file)
                labels_in = pd.read_csv(labels_file)

            elif path_split[-1]=='.parquet':
                data_file = self.data_loc + 'recon3D_d'+k+'.parquet'
                labels_file = self.data_loc + 'labels_d'+k+'.parquet'
                data_in = pd.read_parquet(data_file, engine='fastparquet')
                labels_in = pd.read_parquet(labels_file, engine='fastparquet')

            pt_labels = labels_in['pt'].values.tolist()

            # Updates: Read in the y-local values as well
            y_local = labels_in['y-local'].values.tolist()

            data_in = data_in.to_numpy()
            #TODO: Following only for smart-pixels app. Reshape the data into 2D with timeslices
            data_used = data_in.reshape((data_in.shape[0],20,273))

            del data_in

            N = data_used.shape[0] # total no. of samples to be encoded into spikes
        
            n_features = data_used.shape[2]
            t_steps = data_used.shape[1]
    
            spike_times = []
            for n in range(N):
                spike_times.append([])
                for i in range(n_features):
                    voltage_trace = data_used[n,:,i]
                    if any(voltage_trace):
                        if self.sampling_scale < 200:   #TODO
                            voltage_trace, xscale = self.upsample_trace(voltage_trace)
                            res = int(self.sampling_scale)
                        else:
                            res = 200

                        pos_times, neg_times = self.voltage2time(voltage_trace,res)
                        spike_times[n].append([pos_times, neg_times])
                    else:
                        spike_times[n].append([[],[]])
                        #continue
    
            #Save the spike file to an existing array:
            # TODO: hdf5
            times = np.array(spike_times, dtype=object)

            # TODO:combine pt and y-local
            labels = np.array(pt_labels)
            y_local = np.array(y_local)
            # np.savez(<>, x=labels, y=y_local)

            np.save(self.spike_data_path+'/data_spikes'+k+'.npy', times)
            np.save(self.spike_data_path+'/labels'+k+'.npy', labels)

        return self.spike_data_path, indices


    def voltage2time(self, voltage_trace, res):
        # Input the voltage trace per channel that has all 20 or more timeslices:
        # Convert voltage_trace to spike pulses:
        pulse_times = []
        neg_times = []
        next_step = 0
        k=0
        # NOTE: the first data point starts from 200ps, hence the offset if resolution is below 200ps
        # TODO: Update for downsampling the data to time resolution > 200ps
        if res == 200:
            offset =0
        else:
            offset = 200 - res

        while k < len(voltage_trace):
            if k==0 and voltage_trace[k]>=self.charge_thresh:
                pulse_times.append(res*(k+1)+offset)
                cur_val = voltage_trace[k]
                k = k+1
            elif voltage_trace[k]<self.charge_thresh: 
                cur_val = voltage_trace[k]
                k = k+1
            else:
                #NOTE: Here cur_val is usually vol_trace[k-1] or prev 'k'
                if len(pulse_times)==0:
                    pulse_times.append(res*(k+1)+offset)
                    cur_val = voltage_trace[k]

                next_step = cur_val + self.vdiff
                # NOTE: To encode rising edge
                # Check which of the trace values is the vdiff jump closest to
                closest_v = [j - next_step for j in voltage_trace[k:]]
        
                # NOTE: to encode falling edge
                next_down_step = cur_val - self.vdiff
                nearest_v = [next_down_step - j for j in voltage_trace[k:]]

                for p in range(len(closest_v)):
                    if closest_v[p]>=0:
                        break
                
                for q in range(len(nearest_v)):
                    if nearest_v[q]>=0:
                        break

                if p>q:
                    k = k + q
                    neg_times.append(res*(k+1)+offset)
                elif q>p:
                    k = k+p
                    pulse_times.append(res*(k+1)+offset)

                if p==len(closest_v)-1 and q==len(nearest_v)-1:
                    break

                cur_val = voltage_trace[k]

        return pulse_times, neg_times

    
    def upsample_trace(self, voltage_trace):
	# NOTE: This is based on the current dataset which has samples starting at 200ps
	# and lasting for 4000ps
        res = int((4000-200)/self.sampling_scale + 1)
        x = np.linspace(200,4000,20)
        xscale = np.linspace(200,4000,res)
        upscaled = np.interp(xscale,x,voltage_trace)

        return upscaled, xscale

    # NOTE: This function is called during the training and test phase
    def load_spike_file(self, file_idx, pt_cutoff, only_pos, balanced=True, ret_pt=False):

        data_file = self.spike_data_path+'/data_spikes'+str(file_idx)+'.npy'
        pt_file = self.spike_data_path+'/labels'+str(file_idx)+'.npy'

        X = np.load(data_file, allow_pickle = True)
        pT = np.load(pt_file)

        pT = pT.astype(float)

        pt_bins = [-5, -pt_cutoff, 0, pt_cutoff, 5]   #TODO: Only for smart-pixels, need to be generalized

        y = np.digitize(pT, pt_bins, right=True)
        y = y - 1
        y = y.astype(int)

        if only_pos:
            y[y==2]=0
            y[y==3]=1

            label_indx = [i for i in range(len(y))]
            random.shuffle(label_indx)

            X = X[label_indx]
            y = y[label_indx]
            pT = pT[label_indx]

            if balanced:
                class_0 = np.where(y==0)[0].tolist()
                class_1 = np.where(y==1)[0].tolist()

                if len(class_0)<len(class_1):
                    class_1 = np.random.choice(class_1, (len(class_0),), replace=False).tolist()
                else:
                    class_0 = np.random.choice(class_0, (len(class_1),), replace=False).tolist()

                # Extract balanced class indices
                indices = class_0 + class_1
                X = X[indices]
                y = y[indices]
                pT= pT[indices]
            else:
                class_0 = np.where(y==0)[0].tolist()
                class_1 = np.where(y==1)[0].tolist()
                indices = class_0 + class_1
            
        else:
            y[y==3] = 0

            label_indx = [i for i in range(len(y))]
            random.shuffle(label_indx)

            X = X[label_indx]
            y = y[label_indx]
            pT = pT[label_indx]

            y[y==3] = 0
            class_0 = np.where(y==0)[0].tolist()    # high -ve pt
            class_1 = np.where(y==1)[0].tolist()    # low -ve pt
            class_2 = np.where(y==2)[0].tolist()    # low +ve pt
            #print('Class samples:',len(class_0), len(class_1), len(class_2))
           
            if balanced:
                min_class_size = min(len(class_0), len(class_1), len(class_2))
                class_0 = np.random.choice(class_0, (min_class_size,), replace=False).tolist()
                class_1 = np.random.choice(class_1, (min_class_size,), replace=False).tolist()
                class_2 = np.random.choice(class_2, (min_class_size,), replace=False).tolist()

                indices = class_0 + class_1 + class_2

                X = X[indices]
                y = y[indices]
                pT = pT[indices]
                ylocal = ylocal[indices]

        if ret_pt:
            return X, y, pT 
        else:
            return X, y


    def get_dvs_data(datapath, n):
        import itertools
        
        """
        Function that reads in custom DVS data gathered from the camera.
        The camera generated files are of the form .aedat4
        The data is stored in 'n' sub-directories, where 'n' is the number of categories
        """
        print(os.getcwd())

        orig_path = os.getcwd()
        print('Original path of running',orig_path)

        # Get the names of the sub-directory files:
        file_list = []
        class_list = []
        for category in range(n):
            dirs = datapath + '/'+str(category)+'/'

            # Change directory to the pos-class location:
            os.chdir(dirs)
            #print('Changed directory to:',os.getcwd())

            for file1 in glob.glob('*.aedat4'):
                file_list.append(file1)

            labels_list = list(itertools.chain.from_iterable([n] for i in range(len(file_list))))

            class_list.extend(labels_list)

            # Set working directory to current:
            os.chdir(orig_path)
            print('Set the directory back to:',os.getcwd())
        

        return file_list, class_list

    # Function to read in the DVS data from the aedat4 files:
    # function obtained from INIvation gitlab
    # Source: https://gitlab.com/inivation/dv/dv-python
    def read_dvs(dvs_file):
        from dv import AedatFile
        """
        Returns the dvs event data and the start timestamp of the events recorded
        """

        with AedatFile(dvs_file) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
        
        t_vals=[]
        for e in events:
            t_vals.append(e['timestamp'])
            
        max_t = max(t_vals)
        min_t = min(t_vals)
        
    return events, min_t


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data Encoder for the Fermi lab smart-pixels data")
    parser.add_argument('--data_loc', default='/noback/8hk/Cerndata/ds678/unflipped-positive/', type=str, help='Path where the origianl Fermi data is stored')
    parser.add_argument("--timescale", default=200, type=int, help="Provide the scale with respect to 200ps")
    parser.add_argument("--charge_thresh",default=800.0, type=float, help="threshold above which charge values need to be considered")
    parser.add_argument("--vdiff",default=400.0, type=float, help="voltage difference at which the time pulses will be generated")
    parser.add_argument("--threshold",default=0.2, type=float, help="bin boundaries to divide the data based on pT")
    parser.add_argument("--only_pos", default=True, type=bool, help="if running with only positive half of the dataset.")
    parser.add_argument("--spike_path", default='/noback/8hk/Cerndata/spike_data_ds678', type=str, help="Cached spike data path prefix. NOTE - dO not provide '/' at the end")
    parser.add_argument("--ylocal_min",default=-1.0, type=float, help="range of y-local vlaues to consider for sample")
    parser.add_argument("--ylocal_max",default=1.0, type=float, help="range of y-local vlaues to consider for sample")

    parser.add_argument("--test_split_size", default=0.33, type=float, help="Percentage of test samples for training.")


    args = parser.parse_args()
    data_config = vars(args)

    dataloader = DataLoader(data_config)

    spike_data_path, file_ids = dataloader.convert_to_spikes(check_cached_data=False)

    print('No. of files in the dataset:',len(file_ids))

    n_training = int((1-args.test_split_size)*len(file_ids)) 
    print('No. of training files:',n_training)

    trainfile_ids = file_ids[:n_training]
    testfile_ids = file_ids[n_training:]

    random.shuffle(trainfile_ids)
    
    f = trainfile_ids[0]
    print('File to be read in:',f)

    X, y, pT = dataloader.load_spike_file(f, args.threshold, args.ylocal_max, args.ylocal_min, only_pos = args.only_pos, balanced=True, ret_pt=True)

    print('Input samples:',len(X), ' pt labels:',len(y), len(pT))
