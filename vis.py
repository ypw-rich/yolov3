import cv2 as cv
import numpy as np
import os
from glob import glob
import argparse


def read_xywh(path, sep=' '):
    with open(path) as f:
        s = f.read()
    points = np.array([list(map(float, x[2:].split(sep))) for x in s.strip().split('\n')])
    return points


def read_points(path, sep=' '):
    with open(path) as f:
        s = f.read()
    return np.array([list(map(int, x.split(sep)[:8])) for x in s.strip().split('\n')])


def draw(img, points, color=(0, 255, 0), width=2, ptype='xywh', fw=1, fh=1):
    for p in points:
        if len(p) == 8:
            p = p.reshape(-1, 2)
        else:
            if ptype == 'xywh':
                x, y, ww, hh = p
                h, w, _ = img.shape
                p1 = (x - ww / 2), (y - hh / 2)
                p2 = (x + ww / 2), (y - hh / 2)
                p3 = (x + ww / 2), (y + hh / 2)
                p4 = (x - ww / 2), (y + hh / 2)
                p = np.array([p1, p2, p3, p4])
            else:
                x1, y1, x2, y2 = p
                p1 = x1, y1
                p2 = x2, y1
                p3 = x2, y2
                p4 = x1, y2
                p = np.array([p1, p2, p3, p4])
        p = p.astype(np.float32)
        p[:, 0] /= fw
        p[:, 1] /= fh
        p = p.astype(np.int32)
        cv.drawContours(img, [p], -1, color, width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('--id', type=int)
    parser.add_argument('--image', type=str, default='hanwang')
    parser.add_argument('--pred', type=str, default='data/pred')
    parser.add_argument('--list', type=str, default='data/eval.txt')

    args = parser.parse_args()
    if args.id is None:
        parser.print_help()
        exit(-1)

    print(args)
    image_path = args.image
    pred_path = args.pred
    if args.list:
        with open(args.list) as f:
            paths = f.read().strip().split('\n')
    else:
        paths = glob(f'{pred_path}/*.txt')

    ids = [x.split('/')[-1].split('.')[0] for x in paths]
    pred_paths = [f'{pred_path}/{x}.txt' for x in ids]
    true_paths = [f"{image_path}_train/a/{x}_line.txt" for x in ids]

    pred_path = pred_paths[args.id]
    true_path = true_paths[args.id]
    img_path = f'{image_path}/a/image/{ids[args.id]}.jpg'
    print(img_path)

    img = cv.imread(img_path)
    draw(img, read_points(pred_path), color=(0, 0, 255))
    draw(img, read_points(true_path), color=(0, 255, 0))
    cv.imwrite('a.jpg', img)
    os.system('open a.jpg')
    os.system('xdg-open a.jpg')
    os.system('start a.jpg')
