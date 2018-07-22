#!/usr/bin/env python
'''
script  that calls main step of pipeline
'''
import os.path

# TEMP, add package to your path outside of here
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sillytalks.transform.pipeline

def main():
    sillytalks.transform.pipeline.full_train()
    sillytalks.transform.pipeline.full_test()
    
    sillytalks.transform.pipeline.write_labels()

if __name__=='__main__':
    main()

