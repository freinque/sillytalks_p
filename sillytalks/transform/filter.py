'''
set of filtering tools from scipy in the form of sklearn transforms

using librosa

assuming data samples have varying length, but -1,1 range
'''

import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
import scipy.fftpack

from sklearn.base import TransformerMixin

from scipy.signal import butter, lfilter

import librosa

#import error import matplotlib.pyplot as plt

from ..etc import conf

def butter_bandpass(lowcut, highcut, fs=conf.SAMPLING_RATE, order=5):
    ''' from scipy cookbook
    http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs=conf.SAMPLING_RATE, order=5):
    ''' from scipy cookbook
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class ButterBandpass(TransformerMixin):
    def __init__(self, lowcut, highcut):
        self.lowcut = lowcut
        self.highcut = highcut
    
    def fit(self,X):    
        return self
    
    def transform(self, X):
        return butter_bandpass_filter( X, self.lowcut, self.highcut )



def ft(X, window='hanning', window_length=2048, hop_length=100):
    '''
    basic window-based Fourier transform
    '''
    ft = librosa.core.stft(X, hop_length=hop_length, n_fft=window_length, window=window, pad_mode='reflect')
    
    return np.abs(ft)

def trim_ft(ft, plot=False):
    '''
    '''
    prop_ncst = 40.
    
    ft = pd.DataFrame(ft).T
    ft_t = ft.sum(axis=1)
    ft_t_cumsum = ft_t.cumsum()
    
    if plot:
        print 'time '
        ft_t_cumsum.plot()
        #import error plt.show()
        print 'freq '
        ft.sum(axis=0).cumsum().plot()
        #import error plt.show()

    ft_t_sum = ft_t.sum()
    c_min = ft_t_sum/prop_ncst
    c_max = ft_t_sum-ft_t_sum/prop_ncst

    ft_trimmed = ft[(ft_t_cumsum<c_max) & (ft_t_cumsum>c_min)] # clip on time range
    #print 'ft_trimmed.shape', ft_trimmed.shape
    ft_trimmed = ft_trimmed.iloc[:,:(ft_trimmed.shape[1]*3)/5] # first 3/5 of freq range
    
    return ft_trimmed

class WindowFT(TransformerMixin):
    def __init__(self,  window_length=2048, window='hanning', hop_length=100):
        self.window = window
        self.window_length = window_length
        self.hop_length = hop_length

    def fit(self,X):
        return self
    
    def transform(self, X):
        return ft( X, self.window, self.window_length, self.hop_length )


def resample(ft_trimmed, n_windows=40):
    '''
    reduces the number of samples by a factor windows
    '''
    N_WINDOWS = n_windows
    WIDTH_WINDOWS = max(len(ft_trimmed)/N_WINDOWS, 1)
    ft_trimmed['window'] = range(len(ft_trimmed))
    ft_trimmed['window'] = (ft_trimmed['window']/WIDTH_WINDOWS).astype(int).clip(lower=0, upper=N_WINDOWS)

    window_mean = ft_trimmed.groupby('window').mean()

    window_mean = window_mean/window_mean.max().max()

    if len(window_mean) == 40:
        window_mean.loc[40] = window_mean.loc[39]

    return window_mean


########################################################################################
# scipy signal windows

#get_window(window, Nx[, fftbins])	Return a window.
#barthann(M[, sym])	Return a modified Bartlett-Hann window.
#bartlett(M[, sym])	Return a Bartlett window.
#blackman(M[, sym])	Return a Blackman window.
#blackmanharris(M[, sym])	Return a minimum 4-term Blackman-Harris window.
#bohman(M[, sym])	Return a Bohman window.
#boxcar(M[, sym])	Return a boxcar or rectangular window.
#chebwin(M, at[, sym])	Return a Dolph-Chebyshev window.
#cosine(M[, sym])	Return a window with a simple cosine shape.
#exponential(M[, center, tau, sym])	Return an exponential (or Poisson) window.
#flattop(M[, sym])	Return a flat top window.
#gaussian(M, std[, sym])	Return a Gaussian window.
#general_gaussian(M, p, sig[, sym])	Return a window with a generalized Gaussian shape.
#hamming(M[, sym])	Return a Hamming window.
#hann(M[, sym])	Return a Hann window.
#hanning(M[, sym])	Return a Hann window.
#kaiser(M, beta[, sym])	Return a Kaiser window.
#nuttall(M[, sym])	Return a minimum 4-term Blackman-Harris window according to Nuttall.
#parzen(M[, sym])	Return a Parzen window.
#slepian(M, width[, sym])	Return a digital Slepian (DPSS) window.
#triang(M[, sym])	Return a triangular window.
#tukey(M[, alpha, sym])	Return a Tukey window, also known as a tapered cosine window.


