#!/bin/bash

if [ ! -d "models" ]; then
    if [ ! -f "models.tar.gz" ]; then
        wget https://github.com/wlc952/GPT-SoVITS-TPU/releases/download/v0.2/models.tar.gz
    fi
    tar xzf models.tar.gz
    rm -rf models.tar.gz
fi
