# dilated-cnn-ner

This code implements the models described in the paper
"[Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)"
by [Emma Strubell](https://cs.umass.edu/~strubell), [Patrick Verga](https://cs.umass.edu/~pat),
[David Belanger](https://cs.umass.edu/~belanger) and [Andrew McCallum](https://cs.umass.edu/~mccallum).

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

2. Get some pretrained word embeddings, e.g. [SENNA embeddings](http://ronan.collobert.com/senna/download.html) or
  [Glove embeddings](https://nlp.stanford.edu/projects/glove/). The code expects a space-separated file
  with one word and its embedding per line, e.g.:
   ```
   word 0.45 0.67 0.99 ...
   ```
   Make a directory for the embeddings:
   ```
   mkdir -p data/embeddings
   ```
   and place the file there.

3. Perform all data preprocessing for a given configuration. For example:

  ```
  ./bin/preprocess.sh conf/conll/dilated-cnn.conf
  ```

  This calls `preprocess.py`, which loads the data from text files, maps the tokens, labels and any other features to
  integers, and writes to TensorFlow tfrecords.

Training
----
Once the data preprocessing is completed, you can train a tagger:

  ```
  ./bin/train-cnn.sh conf/conll/dilated-cnn.conf
  ```

Evaluation
----
By default, the trainer will write the model which achieved the best dev F1. To evaluate a saved model on the dev set:

  ```
  ./bin/eval-cnn.sh conf/conll/dilated-cnn.conf --load_model path/to/model
  ```
To evaluate a saved model on the test set:

  ```
  ./bin/eval-cnn.sh conf/conll/dilated-cnn.conf --load_model path/to/model test
  ```


Configs
----
Configuration files (`conf/*`) specify all the data, parameters, etc. for an experiment.
