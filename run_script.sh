#!/bin/bash

## Gypsum run config

#SBATCH --job-name=test
#SBATCH --output=out.txt  # output file
#SBATCH -e res.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to 
#SBATCH --nodes=1 # Number of nodes to use
#SBATCH --ntasks-per-node=8 # Number of cores per node



## STEP 1: Initialize variable -- Assign root directory path to DILATED_CNN_NER_ROOT

#export DILATED_CNN_NER_ROOT="" #path to the root directory containing the source code
#export DATA_DIR=$DILATED_CNN_NER_ROOT/data 
#export PYTHONPATH=$PYTHONPATH:$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/src 
#export models_dir=$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/models/initial_setup



## STEP 2: Create directories

#bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/create_experiments.sh -e=initial_setup



## STEP 3: Parse the raw data

#cd arxiv-metadata-tagger/src/preprocessing
#python -u parse_json_2.py



## STEP 4: Preprocessing 1

#cd arxiv-metadata-tagger/src/preprocessing
#python -u bin_positions_preprocess.py -x 6 -y 8 -i ../../input_file.txt -b output.txt -d ../../../data/initial_setup/arxiv/



## STEP 5: Preprocessing 2

#bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/preprocess.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/log.txt



## STEP 6: Training

#bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/train-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/log_train.txt



## STEP 7: Testing on the dev set

#bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/eval-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf --load_model $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/saved_models/dilated-cnn.tf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/eval_dev.txt



## STEP 8: Testing on the test set

#bash $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/bin/eval-cnn.sh $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/conf/arxiv/dilated-cnn.conf test --load_model $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/saved_models/dilated-cnn.tf >> $DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/results/eval_test.txt



## MISC: Get page geometry

#cd arxiv-metadata-tagger/src/preprocessing
#python -u get_geometry.py


