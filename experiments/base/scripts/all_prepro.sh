set -eu

find ./configs -type f -name "prepro_*.jsonnet" | grep -v "/prepro_config_base.jsonnet$" | sort | xargs -t -I % zsh ./scripts/prepro.sh %

echo "Finish all preprocessing"
