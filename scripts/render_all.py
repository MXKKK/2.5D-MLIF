import sys
sys.path.append('.')
import os

file = open('data/thuman2/all.txt')
cmd = 'export MESA_GL_VERSION_OVERRIDE=3.3'
print(cmd)
os.system(cmd)
while 1:
    line = file.readline()
    if not line:
        break
    cmd = 'python scripts/render_single.py -s ' +  line[:-1] + ' -o data/thuman2_36views_1024 -r 36 -w 1024'
    print(cmd)
    os.system(cmd)
    #break

