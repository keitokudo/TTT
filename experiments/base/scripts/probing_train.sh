set -eux
PROBING_TRAIN_RESULT_FILE_PATH=$1

LOG_DIR=$(dirname $PROBING_TRAIN_RESULT_FILE_PATH)
PROBING_TRAIN_STEM=$(basename $PROBING_TRAIN_RESULT_FILE_PATH .json)
# Remove "result_" prefix
PROBING_TRAIN_STEM=${PROBING_TRAIN_STEM#result_}
MODEL_SAVE_PATH=$LOG_DIR/linear_classifier_model_${PROBING_TRAIN_STEM}

python /project/src/scripts/hidden_states_probing.py \
       --probing_train_result_file_path $PROBING_TRAIN_RESULT_FILE_PATH \
       -m TorchNoneLinearStochasticClassifier \
       -e accuracy \
       -i \
       --model_save_path $MODEL_SAVE_PATH \
       -g 0 1 2 3 0 1 2 3 0 \
       --label_map_path /work/datasets/labels/labels_0_9.json \
       --n_epoch 10000 \
       --n_layer 1 \

# -g 0 1 2 3 0 1 2 3 0 \
