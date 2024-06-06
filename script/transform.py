import os, shutil
import numpy as np
import cv2 as cv
import argparse
from PIL import Image
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, default=True)
    args = parser.parse_args()
    return args


def process(path_list:list, output:str):

    for pa in path_list:
        fi = os.path.basename(pa)
        fpath = pa
        di = fi.split("_")[0]
        dpath = os.path.join(output, di)
        os.makedirs(dpath, exist_ok=True)
        
        if int(di) % 2 == 0:
            shutil.move(fpath, os.path.join(dpath, fi))
        else:
            img = Image.open(fpath)
            # apply flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.save(os.path.join(dpath, fi))


def process_path(path:str, output:str, npro:int=8):
    spath = os.path.join(path, "val", "images")

    path_list = [os.path.join(spath, fi) for fi in os.listdir(spath)]

    num_process = npro
    process_list = []
    for i in range(num_process):
        process_list.append(multiprocessing.Process(target=process, args=(path_list[i::num_process], output)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()


if __name__ == '__main__':

    args = parse_args()

    num_process = 32

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    for s in range(args.split):
        process_path(args.path+f'_{s}', args.output, npro=num_process)
