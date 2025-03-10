#!/bin/sh
srun -p AI4Phys --nodes=1  jupyter notebook  --notebook-dir=/mnt/hwfile/ai4chem/yangzhuo/Faithful-BO/  --ip=0.0.0.0 --port=10049
