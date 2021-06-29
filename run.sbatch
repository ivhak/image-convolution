#!/usr/bin/env bash
#SBATCH --job-name="mi100q_01_nodes_01_tasks"
#SBATCH --partition=mi100q
#SBATCH --time=0-01:00:00
# # SBATCH --gres=gpu:1
#SBATCH --nodelist=n004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=jobs/%j-%x-out.txt
#SBATCH --error=jobs/%j-%x-err.txt

BASE_PATH="${SLURM_SUBMIT_DIR:-/home/iverh/tdt4200_image_convolution}"

KERNELS=(sobelY sobelX laplacian1 laplacian2 laplacian3 gaussian )
BINS=(image_convolution_serial image_convolution_opencl image_convolution_hip)

INPUT_IMAGE="${BASE_PATH}/data/before.bmp"

for bin in ${BINS[*]}; do
    OUTDIR="${BASE_PATH}/jobs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}/${bin}"
    mkdir -p "${OUTDIR}"

    STDERR="${OUTDIR}/stderr.txt"
    LOG_FILE="${OUTDIR}/logs.txt"
    SUM_FILE="${OUTDIR}/sums.txt"
    for k in {0..5}; do
        for i in 5 10 50 100; do
            OUTPUT_IMAGE="${OUTDIR}/${bin}-${KERNELS[$k]}.k-$k.i-$i.bmp"
            srun ${BASE_PATH}/${bin} -i "$i" -k "$k" "${INPUT_IMAGE}" "${OUTPUT_IMAGE}" 2> "${STDERR}" >> "${LOG_FILE}"
            md5sum "${OUTPUT_IMAGE}" >> "${SUM_FILE}"
        done
    done
done
