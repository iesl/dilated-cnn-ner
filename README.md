# dilated-cnn-ner

This code implements the models described in the paper
"[Fast and Accurate Sequence Labeling with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)"
by [Emma Strubell](cs.umass.edu/~strubell), [Patrick Verga](cs.umass.edu/~pat),
[David Belanger](cs.umass.edu/~belanger) and [Andrew McCallum](cs.umass.edu/~mccallum).

Requirements
-----
This code uses TensorFlow >= v1.0 and Python 2.7.

It will probably train on a CPU, but honestly we haven't tried, and highly recommend training on a GPU.


Setup
-----
1. Set up environment variables. For example, from the root directory of this project:

  ```
  export DILATED_CNN_NER_ROOT=`pwd`
  export DATA_DIR=/path/to/conll-2003
  ```

2.

3. Perform all data preprocessing for a given configuration. For example:

  ```
  ./bin/preprocess.sh conf/dilated-cnn.conf
  ```

  This calls `preprocess.py`, which loads the data from text files, maps the tokens, labels and any other features to
  integers, and writes to TensorFlow tfrecords.

Training
----
Once the data preprocessing is completed, you can train a tagger:

  ```
  ./bin/train-cnn.sh conf/dilated-cnn.conf
  ```

Evaluation
----
By default, the trainer will write the model which achieved the best dev F1. To evaluate a saved model on the dev set:

  ```
  ./bin/eval-cnn.sh conf/dilated-cnn.conf --load_model path/to/model
  ```
To evaluate a saved model on the test set:

  ```
  ./bin/eval-cnn.sh conf/dilated-cnn.conf --load_model path/to/model test
  ```


Configs
----
Configuration files (`conf/*`) specify all the data, parameters, etc. for an experiment.
