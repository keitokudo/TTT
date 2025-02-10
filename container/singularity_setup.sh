set -x
source /etc/profile.d/modules.sh
export DEBIAN_FRONTEND="noninteractive"
export TZ="Asia/Tokyo"
export SINGULARITY_TMPDIR=$SGE_LOCALDIR
export SINGULARITY_CACHEDIR=$SGE_LOCALDIR
module load singularitypro
set +x
