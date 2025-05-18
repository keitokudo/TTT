set -eux
THIS_SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# EQUATION_STARTS=(0 7 15 18 25 33 38 45 53)
# EQUATION_ENDS=(6 14 17 24 32 37 44 52 55)

EQUATION_STARTS=(33 25 18 15 7 0)
EQUATION_ENDS=(37 32 24 17 14 6)

NUM_EQUATION_STARTS=${#EQUATION_STARTS[@]}
NUM_EQUATION_ENDS=${#EQUATION_ENDS[@]}
if [ $NUM_EQUATION_STARTS -ne $NUM_EQUATION_ENDS ]; then
    echo "Error: EQUATION_STARTS and EQUATION_ENDS arrays must have the same length."
    exit 1
fi
NUM_EQUATIONS=$NUM_EQUATION_STARTS


cd $THIS_SCRIPT_DIR
zsh env_setup.sh
zsh ./scripts/prepro.sh ./configs/prepro_config_2step_random_edit_teacher_force_eq_1_test.jsonnet

for equation_id in $(seq 0 $((NUM_EQUATIONS - 1))); do
    eq_start=${EQUATION_STARTS[$equation_id]}
    eq_end=${EQUATION_ENDS[$equation_id]}
    echo "Equation ID: $equation_id"
    echo "Equation Start: $eq_start"
    echo "Equation End: $eq_end"
    
    for layer_id in $(seq 0 4 24); do
	export PATCH_START_POSITION=$eq_start
	export PATCH_END_POSITION=$eq_end
	export PATCH_START_LAYER=$layer_id
	export PATCH_END_LAYER=$((layer_id + 3))
	
	zsh ./scripts/test_sliding.sh \
	    ./configs/decode_test_2step_random_edit_teacher_force_sliding_eq_3_config.jsonnet 0 1 2
    done
done
