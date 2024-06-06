import os
import shutil
import cv2 as cv
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    return args


def rename_file(path:str):
    dirs = os.listdir(path)
    for di in dirs:
        dpath = os.path.join(path, di)
        files = os.listdir(dpath)

        for fi in files:
            fpath = os.path.join(dpath, fi)
            fpath2 = os.path.join(dpath, f"{os.path.basename(dpath).split('_')[-1]}_{fi}")

            os.rename(fpath, fpath2)


def split_dir(path:str, num:int=1):

    to_dirs = [os.path.join(os.path.dirname(path), os.path.basename(path)+f"_{i}", "test") for i in range(num)]
    for d in to_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    dpath = os.path.join(path, "test")
    dirs = os.listdir(dpath)

    for i, di in enumerate(dirs):
        path_from = os.path.join(dpath, di)
        path_to = os.path.join(to_dirs[i % num], di)

        if os.path.exists(path_to):
            shutil.rmtree(path_to)
        os.rename(path_from, path_to)


if __name__ == '__main__':

    args = parse_args()
    split_dir(args.path, num=args.split)
