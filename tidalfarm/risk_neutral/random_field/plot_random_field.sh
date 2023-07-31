#!/bin/bash

rm -rf output

python plot_random_field.py

cd output

convert -dispose 2 -delay 80 -loop 0 random_field_seed_*.png random_field.gif

cd ..

