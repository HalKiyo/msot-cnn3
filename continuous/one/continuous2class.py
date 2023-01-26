import bisect
import numpy as np

def convert_EWD(pr_flat, bnd):
    pr_class = np.empty(len(pr_flat))
    for i, value in enumerate(pr_flat):
        label= bisect.bisect(bnd, value) # giving label

