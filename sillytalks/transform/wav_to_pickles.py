import os
import glob
import numpy as np
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack

import pickle

from ..etc import conf

def make_pickles(source_path, target_path, prefix):
    '''
    given 
        folder
    reads wav files in it, normalizes and saves numpy format pickles
    '''
    print 'making pickles out of wav files'
    file_pattern = os.path.join(source_path, '*.wav')
    print 'file_pattern :', file_pattern
    files = glob.glob(file_pattern)

    for wav_file in files:
        print 'processing wav file :', wav_file
        sampling_rate, signal = scipy.io.wavfile.read( wav_file )
        print 'sampling_rate :', sampling_rate

        signal = signal/conf.WAV_NORM_FACTOR #normalize signal to [-1,1]
        
        file_name = os.path.basename(wav_file)
        saved_path = os.path.join(target_path, prefix+'_'+file_name[:-4]+'.p')
        print 'saved_path :', saved_path
        pickle.dump( signal, open( saved_path, "wt" ) )

def process(base_path, target_path):
    '''
    given
        folder
    calls make_pickles on subfolders
    '''
    folders = os.listdir(base_path)

    for folder in folders:
        print 'processing folder :', folder
        
        path = os.path.join(base_path, folder)
        make_pickles(path, target_path, folder)

def process_train():
    '''
    proesses train dataset
    '''
    print 'processing training set'
    base_path = os.path.join(conf.RAW_DATA_PATH, 'train/audio')
    target_path = os.path.join(conf.DATA_PATH,'pickles/train/')

    process(base_path, target_path)
    print 'processed training set'

def process_test():
    '''
    proesses test dataset
    '''
    print 'processing test set'
    base_path = os.path.join(conf.RAW_DATA_PATH, 'train/audio')
    base_path = os.path.join(conf.RAW_DATA_PATH, 'test/audio')
    target_path = os.path.join(conf.DATA_PATH,'pickles/test/')

    process(base_path, target_path)
    print 'processed test set'
