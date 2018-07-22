# sillytalks

**[TensorFlow Sample](https://www.tensorflow.org/versions/master/tutorials/audio_recognition) tentative solution**

### example usage

~~~~
python scripts/prepare_data.py
~~~~

takes .wav data (download from kaggle), normalizes and saves in numpy format

~~~~
python scripts/pipeline_data.py
~~~~

applies series of transformations to data to reach model-ready state

~~~~
python scripts/train.py --modelinit=True --modelname=base
~~~~

initializes new model and go through one training cycle

~~~~
python scripts/train.py --modelinit=False --modelname=base
~~~~

instantiates saved model and go through one training cycle

### NOTES

**Very high level view of approach taken:**

* basic filterng, triming, fourier transform and downsampling of wav samples

* symmetrizing with respects to time shifts, etc.

* training a convolutional neural network on resulting dataset (classifier on 30 labels)

