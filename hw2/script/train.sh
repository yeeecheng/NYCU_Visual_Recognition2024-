#!./bash
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1  main.py --data_dir ./nycu-hw2-data/
# CUDA_VISIBLE_DEVICES="0, 1, 2, 3" torchrun --nproc_per_node=4  main.py --data_dir ./nycu-hw2-data/
