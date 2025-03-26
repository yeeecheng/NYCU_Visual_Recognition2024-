#!./bash
# CUDA_VISIBLE_DEVICES="4" python main.py
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 main.py
