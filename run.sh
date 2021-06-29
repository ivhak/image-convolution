#!/usr/bin/env bash

KERNELS=(sobelY sobelX laplacian1 laplacian2 laplacian3 gaussian )
BIN="$1"

if [[ ! -x $1 ]]; then
    echo "\"$1\" is not an executable file"
fi

OUTDIR="out/${BIN}"

LOGFILE="${OUTDIR}/logs.txt"
SUMFILE="${OUTDIR}/sums.txt"

mkdir -p "${OUTDIR}"
echo "" > "${LOGFILE}"
echo "" > "${SUMFILE}"
for k in {0..5}; do
    for i in 5 10 50 100; do
        FILE="${OUTDIR}/${BIN}-${KERNELS[$k]}.k-$k.i-$i.bmp"
        ./${BIN} -i "$i" -k "$k" data/before.bmp "${FILE}" | tee -a "${LOGFILE}"
        md5sum "${FILE}" >> "${SUMFILE}"
    done
done
