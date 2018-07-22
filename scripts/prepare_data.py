#!/usr/bin/env python
'''
script that calls preparatory step of pipeline
'''
import os.path

# TEMP, add package to your path outside of here
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sillytalks.transform.wav_to_pickles

def main():
    sillytalks.transform.wav_to_pickles.process_train()
    sillytalks.transform.wav_to_pickles.process_test()

if __name__=='__main__':
    main()

