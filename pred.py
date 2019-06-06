import os
import PIL
import torch
from torchvision.transforms.functional import to_tensor
from models import *
from tqdm import tqdm
from glob import glob
import cv2 as cv
from IPython.display import Image, display
from vis import draw, read_points, read_xywh

from yolov3_text_detection.text_detector import TextDetector
from yolov3_text_detection.utils import get_boxes
from yolov3_text_detection.python_nms import nms
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', formatter_class=argparse.MetavarTypeHelpFormatter)
    parser.add_argument('--id', type=int)
    parser.add_argument('--weight', type=str, default='weights/latest.pt')
    parser.add_argument('--cfg', type=str, default='text_binary.cfg')
    parser.add_argument('--list', type=str, default='data/train_small.txt')
    parser.add_argument('--image', type=str, default='hanwang')

    args = parser.parse_args()
    if args.id is None:
        parser.print_help()
        exit(-1)

    print(args)
    yolo_path = args.weight
    device = torch_utils.select_device()

    model = Darknet(cfg=args.cfg, img_size=608).to(device)
    model.eval()
    model.load_state_dict(torch.load(yolo_path, map_location=device)['model'])

    # os.makedirs('data/pred', exist_ok=True)
    with open(args.list) as f:
        paths = f.read().split('\n')

    img_path = paths[args.id]
    id = img_path.split('/')[-1].split('.')[0]
    img_path = f'{args.image}/a/image/{id}.jpg'
    print(img_path)

    image = PIL.Image.open(img_path)
    image = image.convert('RGB')

    width = 608
    w, h = image.size
    if w > h:
        new_h = int(width / w * h / 32) * 32
        new_w = width
    else:
        new_w = int(width / h * w / 32) * 32
        new_h = width

    image = image.resize((new_w, new_h), PIL.Image.ANTIALIAS)
    fw, fh = new_w / w, new_h / h

    image = to_tensor(image)
    image = image.unsqueeze(0)

    inf_out, train_out = model(image.cuda())
    print('inf_out', inf_out.shape)

    pred = inf_out[0]
    pred = pred[pred[:, 4] > 0.05]
    pred[:, :4] = xywh2xyxy(pred[:, :4])
    pred = pred.detach().cpu().numpy()
    pred = pred[nms(pred, threshold=0.5)]
    print('pred', pred.shape)

    img = cv.imread(img_path)
    width = max(1, int(img.shape[0] / 800))
    draw(img, pred[:, :4], color=(0, 0, 255), ptype='xyxy', fw=fw, fh=fh, width=width)
    cv.imwrite('a.jpg', img)
    os.system('xdg-open a.jpg')

    box = pred
    scores = pred[:, 5]
    keep = np.where(scores > 0.05)
    box[:, 0:4][box[:, 0:4] < 0] = 0
    box[:, 0][box[:, 0] >= w] = w - 1
    box[:, 1][box[:, 1] >= h] = h - 1
    box[:, 2][box[:, 2] >= w] = w - 1
    box[:, 3][box[:, 3] >= h] = h - 1

    score = box[:, 5]
    boxes = box[keep[0]]
    scores = scores[keep[0]]

    MAX_HORIZONTAL_GAP: int = 50
    MIN_V_OVERLAPS: float = 0.6
    MIN_SIZE_SIM: float = 0.6
    TEXT_PROPOSALS_MIN_SCORE: float = 0.1
    TEXT_PROPOSALS_NMS_THRESH: float = 0.3
    TEXT_LINE_NMS_THRESH: float = 0.7
    text_detector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], (w, h),
                                 TEXT_PROPOSALS_MIN_SCORE,
                                 TEXT_PROPOSALS_NMS_THRESH, TEXT_LINE_NMS_THRESH)

    text_recs = get_boxes(boxes)
    new_box = []
    rx = 1
    ry = 1
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        new_box.append([x1 * rx, y1 * ry, x2 * rx, y2 * ry, x3 * rx, y3 * ry, x4 * rx, y4 * ry])
    new_box = sorted(new_box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    new_box = np.array(new_box)
    print('new_box', len(new_box))

    img = cv.imread(img_path)
    width = max(1, int(img.shape[0] / 800))
    draw(img, new_box, color=(0, 0, 255), ptype='xyxy', fw=fw, fh=fh, width=width)
    cv.imwrite('b.jpg', img)
    os.system('xdg-open b.jpg')
