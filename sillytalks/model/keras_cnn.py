import numpy as np
import pickle
import os

import glob

from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model, load_model
from keras.layers import Input
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from ..etc import conf

def batch_generator(files, labels, batch_size, num_classes):
    '''
    generator that yields batch for model
    '''
    while True:
        #print 'generator called: '
        X_train = []
        Y_train = []
            # picks random sample in files
        fil = np.random.choice(files, batch_size)
        for f in fil:
            target = os.path.basename(f).split('_')[0]
            feat = pickle.load( open(f, 'rb'))
    
            X_train.append( np.array([ np.array(feat) ]) )
            Y_train.append(target)
            
        X_train = np.array(X_train)
        #print 'X_train.shape', X_train.shape
        
        Y_train = np.array(Y_train)
        Y_train = np.array( [ labels.index(y) for y in Y_train ] )
        Y_train = np_utils.to_categorical(Y_train, num_classes)
        #print 'Y_train.shpe', Y_train.shape
        
        yield X_train, Y_train

def get_model():
    '''
    initial model architecture
    '''
        # define layers
    conv_55_1 = Conv2D(32, (5, 5), padding='same', activation='relu', kernel_constraint=maxnorm(3), data_format="channels_first")
    dropout_20 = Dropout(0.2)
    conv_55_2 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_constraint=maxnorm(3), data_format="channels_first")
    max_pooling = MaxPooling2D(pool_size=(5, 5))

    flatten = Flatten()

    dense_128_1 = Dense(128, activation='relu', kernel_constraint=maxnorm(3))
    dense_128_2 = Dense(128, activation='relu', kernel_constraint=maxnorm(3))
    dropout_50 = Dropout(0.5)
    final_layer = Dense(conf.NUM_CLASSES, activation='softmax')

        # create model instance
    inp = Input(shape=(1,41, 615))
    out = final_layer(dropout_50(dense_128_2(dense_128_1(flatten(max_pooling(conv_55_2(dropout_20(conv_55_1(inp)))))))))

    model = Model(inputs=inp, outputs=out)
        
        # optimizer
    sgd = SGD(lr=conf.OPT_LEARNING_RATE, momentum=0.9, decay=conf.OPT_DECAY, nesterov=False)
        
        # compile model with optimizer
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print 'model summary ', model.summary()
    return model

def init_model(model_name):
    '''
    loads model architecture and saves initial weights
    '''
    model = get_model()

    print 'saving fresh weights ', model_name, ' to ', os.path.join(conf.MODEL_PATH, model_name+'.hdf5')
    #load doesn't work model.save(os.path.join(conf.MODEL_PATH, model_name+'.hdf5'))
    model.save_weights(os.path.join(conf.MODEL_PATH, model_name+'.hdf5'))
    return model

def load_model(model_name):
    '''
    loads model architecture and loads saved weights
    '''
    model = get_model()

    print 'loading weights ', model_name, ' from ', os.path.join(conf.MODEL_PATH, model_name+'.hdf5')
    model.load_weights(os.path.join(conf.MODEL_PATH, model_name+'.hdf5'))
    return model

def train(model, model_name, epochs=conf.EPOCHS, steps_per_epoch=conf.STEPS_PER_EPOCH, validation_steps=conf.VALIDATION_STEPS):
    '''
    fits given model for specified number of steps
    '''
    checkpointer = ModelCheckpoint(filepath=os.path.join(conf.MODEL_PATH, model_name+'.hdf5'), verbose=1, save_best_only=False)

    files = glob.glob(os.path.join(conf.DATA_PATH, 'pickles/pipeline_train/*.p'))
    labels = pickle.load(open(os.path.join(conf.DATA_PATH, 'labels.p'), 'rb'))
    batch_size = conf.BATCH_SIZE

    model.fit_generator(
            batch_generator(files, labels, batch_size, conf.NUM_CLASSES), 
            validation_data = batch_generator(files, labels, batch_size, conf.NUM_CLASSES),
            validation_steps = validation_steps,
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs,
            workers=4, use_multiprocessing=True,
            callbacks=[checkpointer,],
            )

