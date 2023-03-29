import sys
sys.path.append('.')
import os

file = open('data/thuman2/all.txt')
while 1:
    line = file.readline()
    if not line:
        break
    cmd = 'python scripts/vis_single.py -s ' +  line[:-1] + ' -o data/thuman2_36views_1024 -r 36 -m gen'
    print(cmd)
    os.system(cmd)

