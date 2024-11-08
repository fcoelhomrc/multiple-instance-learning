#!/bin/bash
#
#SBATCH --partition=debug_8gb
#SBATCH --qos=debug_8gb


# Commands
python3 datasets/paip.py
