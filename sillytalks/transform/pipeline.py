'''
'''

import os
import glob
import pickle
import numpy as np

from sklearn.base import TransformerMixin

import sillytalks.transform.filter

from ..etc import conf

class Rescaler(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X):    
        self.absmax = abs(X).max()
        return self
    
    def transform(self, X):
        if self.absmax >0:
            return X/self.absmax
        else:
            return X


class SubstractMean(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X):    
        self.mean= X.mean()
        return self
    
    def transform(self, X):
        return X - self.mean

    def inverse_transform(X):
        return X + self.mean


def full(source_path, target_path):
    '''
    given
        folder
    applies entire above pipeline to files
    '''
    files = os.listdir(source_path)
    
    #TEMP for testing purposes
    #files = ['off_c0445658_nohash_4.p',]
    for f in files[:2]:
        file_name = os.path.basename(f)
        saved_path = os.path.join(target_path, file_name[:-2]+'.p')
        if not os.path.exists(saved_path):
            print 'pipelining wav file :', f
            input_file = pickle.load(open( os.path.join(source_path, f), "rt" ))        
            
                ## recenter
            recenter = SubstractMean()
            recentered = recenter.fit_transform(input_file)
            #print recentered.max()
            
                ## rescale
            rescale = Rescaler()
            rescaled = rescale.fit_transform(recentered)
            #print rescaled
                
                ## bandpass filter
            bandpass = sillytalks.transform.filter.ButterBandpass(lowcut=50, highcut=6000)
            filtered = bandpass.fit_transform(rescaled)
            #print filtered
                
                ## stft
            window_transform = sillytalks.transform.filter.WindowFT(window_length=2048, window='hanning', hop_length=25)
            ft = window_transform.fit_transform(filtered)
        
                ## trim
            ft_trimmed = sillytalks.transform.filter.trim_ft(ft)
                
                ## resample
            ft_resampled = sillytalks.transform.filter.resample(ft_trimmed)

            output_file = ft_resampled

            print 'saved to :', saved_path
            pickle.dump( output_file, open( saved_path, "wt" ) )
 
def full_train():
    print 'pipeline called on train'
    source_path = os.path.join(conf.DATA_PATH, 'pickles/train/')
    target_path = os.path.join(conf.DATA_PATH, 'pickles/pipeline_train/')

    full(source_path, target_path)

def full_test():
    print 'pipeline called on test'
    source_path = os.path.join(conf.DATA_PATH, 'pickles/test/')
    target_path = os.path.join(conf.DATA_PATH, 'pickles/pipeline_test/')

    full(source_path, target_path)

def write_labels():
    '''
    generates label file, a python list with all the available labels (words) seen
    '''
    files = glob.glob(os.path.join(conf.DATA_PATH, 'pickles/pipeline_train/*.p'))

    Y_train = []
    for f in files[:1000]:
        target = os.path.basename(f).split('_')[0]
        Y_train.append(target)

    Y_train = np.array(Y_train)
    labels = list(np.sort(np.unique(Y_train)))
    
    print 'writing ', os.path.join(conf.DATA_PATH, 'labels.p')
    pickle.dump( labels, open( os.path.join(conf.DATA_PATH, 'labels.p'), "wt" ) )

