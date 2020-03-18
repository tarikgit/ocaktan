### CORD-19 data analysis ###

import numpy as np
import pandas as pd

import os
import json


count = 0
file_exts = []
for dirname, _, filenames in os.walk('/Users/tarikz/Webpage/data/2020-03-13/'):
    for filename in filenames:
        count += 1
        file_ext = filename.split(".")[-1]
        file_exts.append(file_ext)

file_ext_set = set(file_exts)

print(f"Files: {count}")
print(f"Files extensions: {file_ext_set}\n\n=====================\nFiles extension count:\n=====================")
file_ext_list = list(file_ext_set)
for fe in file_ext_list:
    fe_count = file_exts.count(fe)
    print(f"{fe}: {fe_count}")