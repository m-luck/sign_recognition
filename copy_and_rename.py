from shutil import copyfile
import os 

files = []
for r, d, f in os.walk(sys.argv[1]):
    for file_name in files:
        rel_dir = os.p