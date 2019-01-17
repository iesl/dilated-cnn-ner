#!/bin/bash


export DILATED_CNN_NER_ROOT="" #path to the root directory containing the source code
export DATA_DIR=$DILATED_CNN_NER_ROOT/data 
export PYTHONPATH=$PYTHONPATH:$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/src 
export models_dir=$DILATED_CNN_NER_ROOT/arxiv-metadata-tagger/models/initial_setup
