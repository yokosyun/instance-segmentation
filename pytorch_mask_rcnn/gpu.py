import os
import json
import torch

# __all__ = ["get_gpu_prop", "collect_gpu_info"]

def sort(d, tmp={}):
    for k in sorted(d.keys()):
        if isinstance(d[k], dict):
            tmp[k] = {}
            sort(d[k], tmp[k])
        else:
            tmp[k] = d[k]
    return tmp