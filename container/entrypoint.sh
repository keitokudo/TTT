#! /bin/zsh
set -eu
COMMAND=$@

if [ -d "/.singularity.d" ] && [ ! -e "/.dockerenv" ]; then
    . /.singularity.d/env/10-docker2singularity.sh
fi

LIB_DIR="/project/lib"
cd ${LIB_DIR}/transformers
echo 'Installing transformers...'
pip install --editable .

mkdir -p /work/pretrained_lms
mkdir -p /work/datasets/labels
cd /project
COMMAND_ARRAY=(${(z)COMMAND})
exec "${COMMAND_ARRAY[@]}"
