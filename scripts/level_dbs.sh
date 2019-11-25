#!/usr/bin/env bash

if [ -x "$(command -v $p7z_bin)" ]; then
    # .7z
    wget "https://drive.google.com/uc?export=download&id=1pj7HZOHiZwllOlxBYQaHksmuVHfQc5Ab"
else
    # .tar.gz
    wget "https://drive.google.com/uc?export=download&id=19KMFjeZFIzFPannYSMkLZAHE32YL1Ezc"
fi

