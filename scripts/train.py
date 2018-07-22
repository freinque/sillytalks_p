#!/usr/bin/env python
'''
script that calls fit method sequentially on specified model
'''
import os.path
import argparse

# TEMP, add package to your path outside of here
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import sillytalks.model.keras_cnn

def main():
    parser = argparse.ArgumentParser(description='init and trains model')
    parser.add_argument('--modelinit', type=str, default='False',
                help='whether you want to init a new model or not')
    parser.add_argument('--modelname', type=str, default='base',
                help='model name')
    args = parser.parse_args()
    print 'init model : ', args.modelinit
    print 'model name: ', args.modelname
    
    if args.modelinit=='True':
        model = sillytalks.model.keras_cnn.init_model(args.modelname)
    else:
        model = sillytalks.model.keras_cnn.load_model(args.modelname)

    sillytalks.model.keras_cnn.train(model, args.modelname)

if __name__=='__main__':
    main()

