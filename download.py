from __future__ import print_function, division
import argparse
from os.path import join
import zipfile
import os

import subprocess
from urllib.request import Request, urlopen

def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)

download(".", "church_outdoor", "val")
download(".", "living_room", "val")


with zipfile.ZipFile("./church_outdoor_val_lmdb.zip", 'r') as zip_ref:
    zip_ref.extractall(".")


with zipfile.ZipFile("./living_room_val_lmdb.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

os.remove("./church_outdoor_val_lmdb.zip")
os.remove("./living_room_val_lmdb.zip")
