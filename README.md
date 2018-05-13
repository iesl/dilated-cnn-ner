# Arxiv Metadata tagger

This code uses Iterated Dilated Convolutions (ID-CNNs) to perform a token based classification of metadata artifacts (Title, Authors, Abstract, Affiliations) for arxiv researach papers. 

The implementation of ID-CNNs is borrowed from "[Dilated CNN NER](https://github.com/iesl/dilated-cnn-ner)".

Requirements
-----
This code uses TensorFlow >= v1.0 and Python 2.7.

Setup
-----
1. Create a new experiment: 
  ```
  bash bin/create_experiment.sh -e=<EXPERIMENT_NAME>
  ```
  This copies the source files, creates appropriate folders to store input data, save tensorflow models, and save the results of the experiments. The paths mentioned in the bash script have to be modified for the respective environment.

2. Set up environment variables for the experiment:

  ```
  export DILATED_CNN_NER_ROOT=`pwd`
  export DATA_DIR=/path/to/input/files
  export PYTHONPATH=$PYTHONPATH:/path/to/experiment/project/src/
  export models_dir=/path/to/save/tensorflow/models
  ```

3. Get some pretrained word embeddings, e.g. [SENNA embeddings](http://ronan.collobert.com/senna/download.html) or
  [Glove embeddings](https://nlp.stanford.edu/projects/glove/). The code expects a space-separated file
  with one word and its embedding per line, e.g.:
   ```
   word 0.45 0.67 0.99 ...
   ```
   Make a directory for the embeddings:
   ```
   mkdir -p DATA_DIR/data/embeddings
   ```
   and place the file there.

4. Perform all preprocessing for a given configuration.
  
  4.1. Data Preprocessing - 1:
  ```
  python src/preprocessing/bin_positions_preprocess.py -x <x_bins> -y <y_bins> -i <input_file> -b <binned_output_file> -d <train_test_dir>'
  ```
  Where the input file is in the format (one contiguous group of tokens is a single page of the paper)
  ```
  0:0:<page_width>:<page_height>
  token_1 <top_left_x_position>:<top_left_y_position>:<bottom_right_x_position>:<bottom_right_y_position> * I-<label>
  token_2 <top_left_x_position>:<top_left_y_position>:<bottom_right_x_position>:<bottom_right_y_position> * I-<label>
  ...
  ```
  4.2. Data Preprocessing - 2:
  ```
  bash bin/preprocess.sh conf/arxiv/dilated-cnn.conf &>> /path/to/experiment/results/file
  ```

  This calls `preprocess.py`, which loads the data from input files, maps the tokens, labels and positions 
  integers, and writes to TensorFlow tfrecords.

Training
----
Once the data preprocessing is completed, you can train a tagger:

  ```
  bash bin/train-cnn.sh conf/arxiv/dilated-cnn.conf &>> /path/to/experiment/results/file
  ```

Evaluation
----
By default, the trainer will write the model which achieved the best dev F1. To evaluate a saved model on the dev set:

  ```
  bash bin/eval-cnn.sh conf/arxiv/dilated-cnn.conf --load_model path/to/model &>> /path/to/experiment/results/file
  ```
To evaluate a saved model on the test set:

  ```
  bash bin/eval-cnn.sh conf/arxiv/dilated-cnn.conf --load_model path/to/model test &>> /path/to/experiment/results/file
  ```


Configs
----
Configuration files (`conf/*`) specify all the data, parameters, etc. for an experiment.
