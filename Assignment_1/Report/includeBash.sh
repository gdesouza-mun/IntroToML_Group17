#!/bin/bash

# Configuration
K_VALUES=(20 50 150)
QUESTION="Q1"
OUTPUT="latex_images.txt"

# Clear file
> $OUTPUT

for k in "${K_VALUES[@]}"; do
    printf "\\\includegraphics[width=0.95\\\linewidth]{Graphs/%s_k%s.png}\n" "$QUESTION" "$k" >> $OUTPUT
    printf "\\\captionof{figure}{Nearest Neighbor classifier for k = %s with sDAT region in blue and sNC region colored green, with training and test data overlayed}\n" "$k" >> $OUTPUT
    printf "\\\label{fig:%sk%s}\n" "$QUESTION" "$k" >> $OUTPUT
    printf "\\\vspace{20pt}\n\n" >> $OUTPUT
done

echo "Done! Copy the contents of $OUTPUT into your LaTeX file."
