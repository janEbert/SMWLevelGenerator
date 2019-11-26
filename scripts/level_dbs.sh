#!/usr/bin/env bash

if [ -x "$(command -v $p7z_bin)" ]; then
    # .7z
    wget "https://drive.google.com/uc?export=download&id=1RB8T_fT__vqv8lUPdEfbmbm6EmGR3tpw"
else
    # .tar.gz
    wget "https://drive.google.com/uc?export=download&id=1eFe0ZkYEd-eOw1Vfa4nUlLP-zweadDbO"
fi

