#/bin/bash

EXPERIMENT=""
SOURCE=/path/to/git/repo/folder
TARGET=/path/to/copy/source/files
SAVED_MODELS=/path/to/root/to/save/tensorflow/models
DATA_DIR=/path/to/root/of/folder/to/copy/data
RESULTS_DIR=/path/to/root/of/folder/to/save/results
EMBEDDINGS_PATH=/path/to/pretrained/embeddings/file

for i in "$@"
do
case $i in
    -e=*|--extension=*)
    EXPERIMENT="${i#*=}"
    shift # past argument=value
    ;;
esac
done

if [ "${#EXPERIMENT}" -eq "0" ]; then
        echo "No EXPERIMENT named. Exiting"
        exit 1
fi

echo "EXPERIMENT NAME = ${EXPERIMENT}"

rm -rf ${TARGET}/${EXPERIMENT}
cp -R ${SOURCE} ${TARGET}/${EXPERIMENT}
echo "Created the program source folders at location ${TARGET}/${EXPERIMENT}"

rm -rf ${SAVED_MODELS}/${EXPERIMENT}
mkdir -p ${SAVED_MODELS}/${EXPERIMENT}
echo "Created folders for saving TF models at location ${SAVED_MODELS}/${EXPERIMENT}"

rm -rf "${DATA_DIR}/${EXPERIMENT}"
mkdir -p "${DATA_DIR}/${EXPERIMENT}/arxiv"
echo "Created folder to store train, dev, and test files at location ${DATA_DIR}/${EXPERIMENT}/arxiv"
mkdir -p "${DATA_DIR}/${EXPERIMENT}/data/embeddings"
echo "Created folder to store embeddings for input data at location ${DATA_DIR}/${EXPERIMENT}/data/embeddings"
cp ${EMBEDDINGS_PATH} ${DATA_DIR}/${EXPERIMENT}/data/embeddings/lample-embeddings-pre.txt
echo "Copied the embeddings file to above location"

rm -rf ${RESULTS_DIR}/${EXPERIMENT}
mkdir -p ${RESULTS_DIR}/${EXPERIMENT}
echo "Created folder to store results of experiments at location ${RESULTS_DIR}/${EXPERIMENT}"


