#!/usr/bin/env bash

if [ -x "$(command -v $p7z_bin)" ]; then
    # .7z
    wget "https://drive.google.com/uc?export=download&id=1_7Lz4UVKvtptYcX4PzdbEC3Z1ORnFaY0"
else
    # .tar.gz
    wget "https://drive.google.com/uc?export=download&id=1Ujr7l5lpRCO-EROOqobZyi8C-Eu3U4in"
fi

