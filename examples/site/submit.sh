#!/usr/bin/env bash
# Submit an example to a CPU node and a GPU node of the author's cluster.
#
#     ./submit.sh relax                 # one CPU job and one GPU job
#     ./submit.sh relax --device gpu    # only that one
#     ./submit.sh single-point
#
# The partitions, conda environment, and checkpoint path below belong to one
# cluster, change them for yours. The CPU job goes to katla_short, the GPU job
# to katla_l40s. The katla partitions share the hardware and differ in time
# limit, one hour is plenty here. Writes a batch script to a temp file and
# submits it with sbatch, the Slurm log goes next to the records in
# <example>/output/<cpu|gpu>/.
set -euo pipefail

readonly CONDA_SH="/groups/kemi/liasi/software/build/miniconda3/etc/profile.d/conda.sh"
readonly ENV_NAME="cheac"
# the shared checkpoint, the nodes have no Hugging Face access
readonly CHECKPOINT="/groups/kemi/liasi/share/cheac/checkpoints/uma/uma-s-1p2p1.pt"
# the cluster cores are multithreaded, one serves a CPU run of these small
# systems. The CPU run peaked at 5.4 GB of process memory, the checkpoint is
# 2.3 GB, so the limit is set with margin
readonly CPU_CPUS=1
readonly CPU_MEM="8G"
readonly GPU_CPUS=4
readonly GPU_MEM="16G"
readonly TIME_LIMIT="01:00:00"

usage() {
    echo "usage: $0 single-point|relax [--device cpu|gpu]" >&2
    exit 1
}

write_batch() {
    local batch_file="$1" example="$2" target="$3" device="$4" partition="$5"
    local gpus_line="$6" cpus="$7" mem="$8"
    cat >"${batch_file}" <<!EOSBATCH
#!/usr/bin/env bash
#SBATCH --job-name=uma-record-${example}-${target}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${partition}
#SBATCH --output=output/${target}/slurm.log
${gpus_line}

module purge
. "${CONDA_SH}"
conda activate ${ENV_NAME}
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
cd "\${SLURM_SUBMIT_DIR}"

python run.py --device ${device} --checkpoint "${CHECKPOINT}"
!EOSBATCH
}

submit() {
    local example="$1" target="$2"
    local device partition gpus_line cpus mem
    case "${target}" in
        cpu)
            device="cpu"
            partition="katla_short"
            gpus_line=""
            cpus="${CPU_CPUS}"
            mem="${CPU_MEM}"
            ;;
        gpu)
            device="cuda"
            partition="katla_l40s"
            gpus_line="#SBATCH --gpus=1"
            cpus="${GPU_CPUS}"
            mem="${GPU_MEM}"
            ;;
    esac
    mkdir -p "output/${target}"

    local batch_file
    batch_file="$(mktemp)"
    write_batch "${batch_file}" "${example}" "${target}" "${device}" \
        "${partition}" "${gpus_line}" "${cpus}" "${mem}"
    sbatch "${batch_file}"
    rm -f "${batch_file}"
}

main() {
    [[ $# -ge 1 ]] || usage
    local example="$1"
    shift
    [[ "${example}" =~ ^(single-point|relax)$ ]] || usage
    local targets=(cpu gpu)
    case "$#" in
        0) ;;
        2)
            [[ "$1" == --device && "$2" =~ ^(cpu|gpu)$ ]] || usage
            targets=("$2")
            ;;
        *) usage ;;
    esac
    # the batch script runs from the example directory, as a hand run would
    cd "$(dirname "${BASH_SOURCE[0]}")/../${example}"
    command -v sbatch >/dev/null || {
        echo "sbatch not found" >&2
        exit 1
    }
    local target
    for target in "${targets[@]}"; do
        submit "${example}" "${target}"
    done
}

main "$@"
