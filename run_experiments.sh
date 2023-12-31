#!/bin/bash

# dvc exp run --name "search-grid" --queue \
#     -S 'prepare.gamma=0.05,0.1,0.4,0.67' \
#     -S 'prepare.adjust_images_brightness_strategy=None,All,Dark' \
#     -S 'prepare.percentile=25,37.5,50' \
#     -S 'train.epochs=100,250,500,1000' \
#     -S 'train.init_lr=0.0001,0.001,0.01' \
#     -S 'train.patience=10,15,25,50'

dvc exp clean
dvc exp remove --all

dvc exp run -f --name "search-grid" --queue \
    -S 'prepare.gamma=0.4,0.67' \
    -S 'prepare.adjust_images_brightness_strategy=Dark' \
    -S 'prepare.percentile=25' \
    -S 'train.epochs=250' \
    -S 'train.init_lr=0.0001' \
    -S 'train.patience=10'

dvc queue start