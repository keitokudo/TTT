set -eux
THIS_SCRIPT_DIR=$(dirname $(realpath $0))
PROBING_TRAIN_RESULT_FILE_PATH=$1
TEST_FILE_PATH=$2

if [ -z "$PROBING_TRAIN_RESULT_FILE_PATH" ]; then
    echo "PROBING_TRAIN_RESULT_FILE_PATH is not set"
    exit 1
fi

if [ -z "$TEST_FILE_PATH" ]; then
    echo "TEST_FILE_PATH is not set"
    exit 1
fi


zsh $THIS_SCRIPT_DIR/probing_train.sh $PROBING_TRAIN_RESULT_FILE_PATH

LOG_DIR=$(dirname $PROBING_TRAIN_RESULT_FILE_PATH)
PROBING_TRAIN_STEM=$(basename $PROBING_TRAIN_RESULT_FILE_PATH .json)
# Remove "result_" prefix
PROBING_TRAIN_STEM=${PROBING_TRAIN_STEM#result_}
MODEL_SAVE_PATH=$LOG_DIR/linear_classifier_model_${PROBING_TRAIN_STEM}
zsh $THIS_SCRIPT_DIR/probing_test.sh $MODEL_SAVE_PATH $TEST_FILE_PATH
