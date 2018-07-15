import os
import glob
import numpy as np
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack

import pickle

def make_pickles(source_path, target_path, prefix):
    '''
    '''
    print 'making pickles out of wav files'
    file_pattern = os.path.join(source_path, '*.wav')
    print 'file_pattern :', file_pattern
    files = glob.glob(file_pattern)

    for wav_file in files:
        print 'processing wav file :', wav_file
        sampling_rate, signal = scipy.io.wavfile.read( wav_file )
        print 'sampling_rate :', sampling_rate

        signal = signal/32768.0 #normalize signal to [-1,1]
        
        file_name = os.path.basename(wav_file)
        saved_path = os.path.join(target_path, prefix+'_'+file_name[:-4]+'.p')
        print 'saved_path :', saved_path
        pickle.dump( signal, open( saved_path, "wt" ) )

def process(base_path, target_path):
    '''
    '''
    folders = os.listdir(base_path)

    for folder in folders:
        print 'processing folder :', folder
        
        path = os.path.join(base_path, folder)
        make_pickles(path, target_path, folder)

