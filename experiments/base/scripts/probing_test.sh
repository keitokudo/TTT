set -eux

MODEL_LOAD_PATH=$1
TEST_FILE_PATH=$2

LOG_DIR=$(dirname $TEST_FILE_PATH)
TEST_STEM=$(basename $TEST_FILE_PATH .json)
# Remove "result_" prefix
TEST_STEM=${TEST_STEM#result_}

MODEL_LOAD_PATH_STEM=$(basename $MODEL_LOAD_PATH)
# Remove "linear_classifier_model_" prefix
MODEL_LOAD_PATH_STEM=${MODEL_LOAD_PATH_STEM#linear_classifier_model_}
OUTPUT_FILE_PATH=$LOG_DIR/linear_classifier_result_${MODEL_LOAD_PATH_STEM}_${TEST_STEM}.jsonl

python /project/src/scripts/hidden_states_probing.py \
       --probing_test_result_file_path $TEST_FILE_PATH \
       -m TorchNoneLinearStochasticClassifier \
       -e accuracy \
       -i \
       --model_load_path $MODEL_LOAD_PATH \
       -g 0 1 2 3 0 1 2 3 0 \
       --label_map_path /work/datasets/labels/labels_0_9.json \
       --n_epoch 10000 \
       --n_layer 1 \
       --ignore_correctness \
       -o $OUTPUT_FILE_PATH

echo "Output written to $OUTPUT_FILE_PATH"

# -g 0 0 0 0 0 0 0 0 0
# -g 0 1 2 3 0 1 2 3 0
