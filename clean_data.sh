#!/bin/bash

BASE_DIR="ai4mars-dataset-merged-0.6/msl"
LABEL_DIR="${BASE_DIR}/ncam/labels/test/masked-gold-min3-100agree"
NCAM_DIR="${BASE_DIR}/ncam/images/edr"

OUT_BASE="AI4Mars_Data/msl_nav"
mkdir -p "$OUT_BASE"

OUT_IMG="${OUT_BASE}/img_test"
OUT_LABEL="${OUT_BASE}/labels_test"
#OUT_UNLABELED="${OUT_BASE}/unlabeled_train"

mkdir -p "$OUT_IMG"
mkdir -p "$OUT_LABEL"
#mkdir -p "$OUT_UNLABELED"

for file in "$NCAM_DIR"/*; do
    filename=$(basename "$file")
    short_name="${filename%.JP*}"

    #echo "$short_name"

    match_found=$(find "$LABEL_DIR" -maxdepth 1 -name "*${short_name}*" | head -n 1)

    if [ "${#match_found}" -gt "0" ]; then

        cp "$match_found" "${OUT_LABEL}/${short_name}.png"
        cp "$file" "${OUT_IMG}/${short_name}.jpeg"
        echo "${OUT_LABEL}/${short_name}.png"
    else
        #cp "$file" "${OUT_UNLABELED}/${short_name}.jpeg"
        continue
    fi

done
