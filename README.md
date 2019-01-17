# Arxiv Metadata tagger

This code uses Iterated Dilated Convolutions (ID-CNNs) to perform a token based classification of metadata artifacts (Title, Authors, Abstract, Affiliations) for arxiv researach papers. 

The implementation of ID-CNNs is borrowed from "[Dilated CNN NER](https://github.com/iesl/dilated-cnn-ner)".

Requirements
-----
This code uses TensorFlow >= v1.0 and Python 2.7.

Setup
-----

1. Set up environment variables for the experiment:

  ```
  export DILATED_CNN_NER_ROOT="" #path to the root directory containing the source code
  export DATA_DIR=$DILATED_CNN_NER_ROOT/data 
  export PYTHONPATH=$PYTHONPATH:$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/src 
  export models_dir=$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/models/initial_setup

  ```
2. Create a new experiment: 
  ```
  bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/create_experiments.sh -e=initial_setup
  ```
  This copies the source files, creates appropriate folders to store input data, save tensorflow models, and save the results of the experiments. The paths mentioned in the bash script have to be modified for the respective environment.



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


4. Parse the raw data file to get input_file.txt.

  ```
  cd arxiv-metadata-tagger/src/preprocessing
  python -u parse_json_2.py
  ```
  
  input_file.txt will be of the following form:
  
  ```
  0:0:<page_width>:<page_height>
  token_1 <top_left_x_position>:<top_left_y_position>:<bottom_right_x_position>:<bottom_right_y_position> * I-<label>
  token_2 <top_left_x_position>:<top_left_y_position>:<bottom_right_x_position>:<bottom_right_y_position> * I-<label>
  ```

5. Perform all preprocessing for a given configuration.
  
  5.1. Data Preprocessing - 1:
  ```
  cd arxiv-metadata-tagger/src/preprocessing
  python -u bin_positions_preprocess.py -x 6 -y 8 -i ../../input_file.txt -b output.txt -d ../../../data/initial_setup/arxiv/
  ```
  The output of this - output.txt will contain the binned positions of the tokens in the document.
  
  5.2. Data Preprocessing - 2:
  ```
  bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/preprocess.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/log.txt
  ```

  This calls `preprocess.py`, which loads the data from input files, maps the tokens, labels and positions 
  integers, and writes to TensorFlow tfrecords.

6. Training

  Once the data preprocessing is completed, you can train a tagger:

  ```
  bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/train-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/log_train.txt
  ```

7. Evaluation

  By default, the trainer will write the model which achieved the best dev F1. 

  To evaluate a saved model on the dev set:

  ```
  bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/eval-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf --load_model $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/saved_models/dilated-cnn.tf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/eval_dev.txt
  ```
  To evaluate a saved model on the test set:

  ```
  bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/eval-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf test --load_model $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/saved_models/dilated-cnn.tf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/eval_test.txt
  ```


Configs
----
Configuration files (`conf/*`) specify all the data, parameters, etc. for an experiment.
