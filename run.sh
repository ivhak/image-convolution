#!/usr/bin/env bash

set -o errexit

KERNELS=(sobelY sobelX laplacian1 laplacian2 laplacian3 gaussian )
# BINS=("image_convolution_opencl" "image_convolution_hip")
BINS=("image_convolution_hip")
INPUT_IMAGE=./data/before.bmp

for bin in ${BINS[*]}; do
    OUTDIR="out/${bin}"
    mkdir -p "${OUTDIR}"

    LOGFILE="${OUTDIR}/logs.txt"
    SUMFILE="${OUTDIR}/sums.txt"

    echo "" > "${LOGFILE}"
    echo "" > "${SUMFILE}"
    echo "--------- $bin"
    for k in {0..5}; do
        for i in 5 10 50 100; do
            FILE="${OUTDIR}/${bin}-${KERNELS[$k]}.k-$k.i-$i.bmp"
            ./${bin} -i "$i" -k "$k" "${INPUT_IMAGE}" "${FILE}" | tee -a "${LOGFILE}"
            md5sum "${FILE}" >> "${SUMFILE}"
        done
    done
done
