#!/usr/bin/env bash
mkdir -p data/train
mkdir -p data/val
mkdir -p checkpoints
unzip train.zip -d data/train
unzip val.zip -d data/val
