#!./bash
# CUDA_VISIBLE_DEVICES="4" python main.py
CUDA_VISIBLE_DEVICES="1,3" torchrun --nproc_per_node=2 main.py
