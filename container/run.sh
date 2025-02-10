COMMAND=$@
set -e

if [ -z $COMMAND ]; then
    echo "COMMAND is empty"
    COMMAND="bash"
fi

THIS_SCRIPT_DIR=$(cd $(dirname $0); pwd)
CODE_DIR=$(readlink -f $THIS_SCRIPT_DIR/..)
SIF_FILE_PATH=$THIS_SCRIPT_DIR/dentaku_probing.sif


if [ -z $WORK_DIR ]; then
    echo "WORK_DIR is empty... Please specify the WORK_DIR by export WORK_DIR=/path/to/your/workdir"
    exit 1
fi
mkdir -p $WORK_DIR

singularity run \
	    --nv \
	    --cleanenv \
	    --home $THIS_SCRIPT_DIR/container_home:/container_home \
	    --env NHOSTS=$NHOSTS \
	    --bind $WORK_DIR:/work,$CODE_DIR:/project \
	    $SIF_FILE_PATH \
	    $COMMAND
