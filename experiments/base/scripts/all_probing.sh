set -eu
LOG_DIR=$1
source ./scripts/setup.sh

COMMAND_FILE_PATH=./commands_probing.sh

python $SOURCE_DIR/scripts/print_all_probing_commnads.py $1 > $COMMAND_FILE_PATH

source $COMMAND_FILE_PATH
echo "Finish all probing.sh"
