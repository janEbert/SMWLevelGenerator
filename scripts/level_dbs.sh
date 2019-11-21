#!/usr/bin/env bash

if [ -x "$(command -v $p7z_bin)" ]; then
    # .7z
    wget "https://drive.google.com/uc?export=download&id=1z1-jS39eFMoHZxlUnOCaHkmjjeIBa-TY"
else
    # .tar.gz
    wget "https://drive.google.com/uc?export=download&id=1tx3g1b8gJ4Kzqq0kDHe8DCoOhhCp8Qq3"
fi

