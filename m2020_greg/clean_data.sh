#!/bin/bash

LABEL_DIR="NAV_Labels"
NCAM_DIR="ncam"
OUT_IMG="images"
OUT_LABEL="labels"
OUT_UNLABELED="unlabeled"

for file in "$NCAM_DIR"/*; do
    filename=$(basename "$file")
    short_name="${filename%.jp*}"

    #echo "$short_name"

    match_found=$(find "$LABEL_DIR" -maxdepth 1 -name "*${short_name}*" | head -n 1)

    if [ "${#match_found}" -gt "0" ]; then

        cp "$match_found" "${OUT_LABEL}/${short_name}.png"
        cp "$file" "${OUT_IMG}/${short_name}.jpeg"
        #echo "${OUT_LABEL}/${short_name}.png"
    else
        cp "$file" "${OUT_UNLABELED}/${short_name}.jpeg"
        #continue
    fi

done
