# generate palm print training data with bezier curves
import bezier
import shutil
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import math

from multiprocessing import Pool

import os, sys, argparse, glob, cv2, random, time
from os.path import join, split, isdir, isfile, dirname
from copy import copy
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

AXIS = [[-0.8, -0.2, 0.2, 1.2],
        [2.6, 3.2, 1.45, 1.5],
        [2.2, 2.4, 1.5, 1.55]]

TLIST = [[0.40, 0.60],
         [0.45, 0.55],
         [0.45, 0.55]]

SLIST = [[-0.05, 0.4],
         [0.05, 0.25],
         [0.05, 0.15]]

SHOW_S = [[0.0, 0.01],
          [0.0, 0.2],
          [0.0, 0.2]]

SHOW_E = [[0.5, 1.0],
          [0.85, 1.0],
          [0.90, 1.0]]

IS_BLACK = True

BG_EN = False

SNUM = [4, 7]

UNI_EN = True

# UNI_NUM = [15, 20]
UNI_NUM = [3, 12]

SLEN = [0.1, 0.50]

CHANGE_EN = True

BLUR_EN = False

BLUR_FIN_EN = False

blur_num = 3

FWIDTH = 1.5
SWIDTH = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--num_ids', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--imsize', type=int, default=256)  # 286
    parser.add_argument('--imagenet', type=str, default='/dockerdata/home/rayshen/data/ILSVRC2012')
    parser.add_argument('--perspective', type=float, default=0, help='probability of performing perspective transform')
    parser.add_argument('--output', type=str, default='./bezier_2000/')
    args = parser.parse_args()
    assert args.num_ids % args.nproc == 0
    return args


def wrap_points(points, M):
    assert isinstance(points, np.ndarray)
    assert isinstance(M, np.ndarray)
    n = points.shape[0]
    augmented_points = np.concatenate((points, np.ones((n, 1))), axis=1).astype(points.dtype)
    points = (M @ augmented_points.T).T
    points = points / points[:, -1].reshape(-1, 1)
    return points[:, :2]


def sample_edge(low, high):
    """
    sample points on edges of a unit square
    """
    offset = min(low, high)
    low, high = map(lambda x: x - offset, [low, high])
    t = np.random.uniform(low, high) + offset

    if t >= 4:
        t = t % 4
    if t < 0:
        t = t + 4

    if t <= 1:
        x, y = t, 0
    elif 1 < t <= 2:
        x, y = 1, t - 1
    elif 2 < t <= 3:
        x, y = 3 - t, 1
    else:
        x, y = 0, 4 - t
    return np.array([x, y]), t


def control_point(head, tail, t=0.5, s=0):
    head = np.array(head)
    tail = np.array(tail)
    l = np.sqrt(((head - tail) ** 2).sum())
    assert head.size == 2 and tail.size == 2
    assert l >= 0
    c = head * t + (1 - t) * tail
    x, y = head - tail
    v = np.array([-y, x])
    v /= max(np.sqrt((v ** 2).sum()), 1e-6)
    return c + s * l * v


def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f' % s
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)


def generate_parameters():
    # head coordinates
    head1, thead1 = sample_edge(AXIS[0][0], AXIS[0][1])
    head2, thead2 = sample_edge(AXIS[1][0], AXIS[1][1])
    head3, thead3 = sample_edge(AXIS[2][0], AXIS[2][1])
    # head4, thead4 = sample_edge(1, 2)

    # tail coordinates
    tail1, ttail1 = sample_edge(AXIS[0][2], AXIS[0][3])
    tail2, ttail2 = sample_edge(AXIS[1][2], AXIS[1][3])
    tail3, ttail3 = sample_edge(AXIS[2][2], AXIS[2][3])
    # if thead4 >= 1.5:
    #     tail4, t = sample_edge(2.5, 3)
    # else:
    #     tail4, t = sample_edge(2, 3)

    c1 = control_point(head1, tail1, t=np.random.uniform(TLIST[0][0], TLIST[0][1]),
                       s=-np.random.uniform(SLIST[0][0], SLIST[0][1]))
    c2 = control_point(head2, tail2, t=np.random.uniform(TLIST[1][0], TLIST[1][1]),
                       s=np.random.uniform(SLIST[1][0], SLIST[1][1]))
    c3 = control_point(head3, tail3, t=np.random.uniform(TLIST[2][0], TLIST[2][1]),
                       s=np.random.uniform(SLIST[2][0], SLIST[2][1]))
    # c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack(
        (head3, c3, tail3))  # , np.vstack((head4, c4, tail4))


def batch_process(proc_index, ranges, args, imagenet_images=None):
    ids_per_proc = int(args.num_ids / args.nproc)
    EPS = 1e-2

    np.random.seed(proc_index)
    random.seed(proc_index)

    #index_file = open(join(args.output, '%.3d-of-%.3d.txt' % (proc_index, args.nproc)), 'w')

    samples_per_proc = ids_per_proc * args.samples

    # average_meter = AverageMeter(name='time')

    local_idx = 0
    for id_idx, i in enumerate(range(*ranges[proc_index])):

        tic = time.time()

        # start/end points of main creases
        nodes1 = generate_parameters()
        start1 = [np.random.uniform(SHOW_S[0][0], SHOW_S[0][1]), np.random.uniform(SHOW_S[1][0], SHOW_S[1][1]),
                  np.random.uniform(SHOW_S[2][0], SHOW_S[2][1])]
        end1 = [np.random.uniform(SHOW_E[0][0], SHOW_E[0][1]), np.random.uniform(SHOW_E[1][0], SHOW_E[1][1]),
                np.random.uniform(SHOW_E[2][0], SHOW_E[2][1])]
        flag1 = [1, 1, 1]
        # flag1 = [np.random.uniform()>0.01, np.random.uniform()>0.01, np.random.uniform()>0.01, np.random.uniform()>0.9]

        # start/end points of secondary creases
        if UNI_EN:
            n2 = np.random.randint(UNI_NUM[0], UNI_NUM[1])
            coord2 = np.random.uniform(0, args.imsize, size=(n2, 2, 2))

            for k in range(n2):
                r = np.random.uniform(0.0, 2 * np.pi)
                delta = np.array([np.cos(r), np.sin(r)]) * np.random.uniform(SLEN[0], SLEN[1]) * args.imsize
                coord2[k][1] = coord2[k][0] + delta

            s2 = np.clip(np.random.normal(scale=0.4, size=(n2,)), -0.6, 0.6)
            t2 = np.clip(np.random.normal(loc=0.5, scale=0.4, size=(n2,)), 0.3, 0.7)

        else:
            n2 = np.random.randint(SNUM[0], SNUM[1])
            coord2 = np.random.uniform(0, args.imsize, size=(n2, 2, 2))
            s2 = np.clip(np.random.normal(scale=0.4, size=(n2,)), -0.6, 0.6)
            t2 = np.clip(np.random.normal(loc=0.5, scale=0.4, size=(n2,)), 0.3, 0.7)

        # synthesize samples for each ID
        for s in range(args.samples):
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((args.imsize + EPS) / dpi, (args.imsize + EPS) / dpi)
            # remove white edges by set subplot margin
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.set_xlim(0, args.imsize)
            ax.set_ylim(args.imsize, 0)
            ax.axis('off')

            perspective_mat = None

            global_idx = samples_per_proc * proc_index + local_idx
            if imagenet_images is not None:
                bg = imagenet_images[global_idx % len(imagenet_images)]
                bg_id = bg['label']
                bg_im = np.array(Image.open(bg['filename']).resize(size=(args.imsize,) * 2))

                if BLUR_EN:
                    if np.random.uniform() >= 0.1:
                        kernel_size = (random.randint(0, 7) * 2 + 1,) * 2
                        bg_im = cv2.blur(bg_im, ksize=kernel_size)
            else:
                if BG_EN:
                    bg_im = np.random.normal(loc=0.0, size=(args.imsize, args.imsize, 3)) + np.random.uniform(
                        size=(1, 1, 3))
                    # bg_im = np.ones((args.imsize, args.imsize, 3))  + np.random.uniform(size=(1, 1, 3))
                    bg_im = np.clip(bg_im, 0.0, 1.0) * 255
                    bg_im = bg_im.astype(np.uint8)
                else:
                    bg_im = np.ones((args.imsize, args.imsize, 3)) * 255
                    bg_im = bg_im.astype(np.uint8)
                bg_id = -1
                bg = {'filename': 'none'}

            bg_im = Image.fromarray(bg_im)
            ax.imshow(bg_im)

            # main creases
            if CHANGE_EN:
                curves1 = [bezier.Curve(n.T * args.imsize, degree=2) for n in nodes1]
                points1 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves1, start1, end1)]
            else:
                curves1 = [bezier.Curve(n.T * args.imsize, degree=2) for n in nodes1]
                points1 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves1, start1, end1)]

            # perspective transformations
            if perspective_mat is not None:
                points1 = [wrap_points(p, perspective_mat) for p in points1]

            paths1 = [Path(p) for p in points1]
            lw1 = np.random.uniform(2.0, 3.0) * FWIDTH
            ecol = np.array([0.0, 0.0, 0.0]) if IS_BLACK else np.random.uniform(0, 0.4, 3)
            patches1 = [patches.PathPatch(p, edgecolor=ecol, facecolor='none', lw=lw1) for p in paths1]
            for p, f in zip(patches1, flag1):
                if f:
                    ax.add_patch(p)

            # secondary creases
            # add turbulence to each sample
            if not CHANGE_EN:
                coord2_ = coord2
                s2_ = s2
                t2_ = t2
            else:
                coord2_ = coord2 + np.random.uniform(-5, 5, coord2.shape)
                s2_ = s2 + np.random.uniform(-0.1, 0.1, s2.shape)
                t2_ = t2 + np.random.uniform(-0.05, 0.05, s2.shape)

            lw2 = np.random.uniform(0.9, 1.1) * SWIDTH
            for j in range(n2):
                points2 = get_bezier(coord2_[j, 0], coord2_[j, 1], t=t2_[j], s=s2_[j]).evaluate_multi(
                    np.linspace(0, 1, 50)).T
                if perspective_mat is not None:
                    points2 = wrap_points(points2, perspective_mat)
                ecol = np.array([0.0, 0.0, 0.0]) if IS_BLACK else np.random.uniform(0, 0.4, 3)
                p = patches.PathPatch(Path(points2), edgecolor=ecol, facecolor='none', lw=lw2)
                ax.add_patch(p)

            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')
            img_rgba = buffer.reshape(args.imsize, args.imsize, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')
            # img = mmcv.rgb2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if BLUR_FIN_EN:
                kernel_size = (blur_num,) * 2
                img = cv2.blur(img, ksize=kernel_size)

            # filename = join(args.output, '%.5d' % i, '%.3d.png' % s)
            filename = join(args.output, f"{i:05}", f"{i:05}_{s:03}.png")
            os.makedirs(dirname(filename), exist_ok=True)
            # mmcv.imwrite(img, filename)
            cv2.imwrite(filename, img)
            plt.close()

            local_idx += 1

        toc = time.time()
        # average_meter.update(toc-tic)
        # print("proc[%.3d/%.3d] id=%.4d [%.4d/%.4d]  (%.3f sec per id)" % (proc_index, args.nproc, i, id_idx, ids_per_proc, average_meter.avg))
        print("proc[%.3d/%.3d] id=%.4d [%.4d/%.4d]" % (proc_index, args.nproc, i, id_idx, ids_per_proc))


if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    spacing = np.linspace(0, args.num_ids, args.nproc + 1).astype(int)

    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    imagenet_images = None

    argins = []
    for p in range(args.nproc):
        argins.append([p, ranges, args, imagenet_images])

    with Pool() as pool:
        pool.starmap(batch_process, argins)

