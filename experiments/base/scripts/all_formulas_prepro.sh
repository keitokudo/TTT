set -eu

find ./configs -type f | grep "_train.jsonnet$" | grep -v "_probing_train.jsonnet$" | sort | xargs -t -I % zsh ./scripts/prepro.sh %
find ./configs -type f | grep "_valid.jsonnet$" | sort | xargs -t -I % zsh ./scripts/prepro.sh %
find ./configs -type f | grep "_test.jsonnet$" | sort | xargs -t -I % zsh ./scripts/prepro.sh %
find ./configs -type f | grep "_probing_train.jsonnet$" | sort | xargs -t -I % zsh ./scripts/prepro.sh %


echo "Finish all preprocessing"


