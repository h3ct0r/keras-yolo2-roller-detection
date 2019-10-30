#! /usr/bin/env python

import argparse
import os
import cv2
import json
import pathlib

import pandas as pd

from lxml import etree
from tqdm import tqdm
from frontend import YOLO


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser(
        description='Train and validate YOLO_v2 model on any dataset')
    parser.add_argument(
        '-ia', '--input_annotation',
        type=str,
        required=True,
        help='The input folder containing annotation files')
    parser.add_argument(
        '-ii', '--input_images',
        type=str,
        required=True,
        help='The input folder containing image files')
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='The output folder')
    parser.add_argument(
        '-c',
        '--conf',
        help='path to configuration file')
    parser.add_argument(
        '-w',
        '--weights',
        help='path to pretrained weights')
    args = parser.parse_args()
    evaluate(
        pathlib.Path(args.input_annotation),
        pathlib.Path(args.input_images),
        pathlib.Path(args.output),
        pathlib.Path(args.conf),
        pathlib.Path(args.weights))


def stratify(row):
    if row.object_count == 0:
        return 0
    elif row.object_count > 0 and row.object_count <= 5:
        return 1
    elif row.object_count > 5 and row.object_count <= 10:
        return 2
    elif row.object_count > 10 and row.object_count <= 15:
        return 3
    elif row.object_count > 15:
        return 4


def read_annotation(input_ann_path):
    info = {
        'image': [],
        'annotation': [],
        'object_count': [],
    }
    xml_files = input_ann_path.glob('*.xml')
    for filename in tqdm(xml_files):
        with open(filename, 'r') as f:
            data = f.read()
        ann = etree.fromstring(data)
        info['image'].append(ann.find('filename').text)
        info['annotation'].append(os.path.basename(filename))
        object_count = len(ann.findall('object'))
        info['object_count'].append(object_count)
    df = pd.DataFrame(info)
    df['stratus'] = df.apply(stratify, axis=1)
    return df


def obj2txt(obj):
    label = obj.find('name').text
    bb = obj.find('bndbox')
    xmin = bb.find('xmin').text
    ymin = bb.find('ymin').text
    xmax = bb.find('xmax').text
    ymax = bb.find('ymax').text
    txt = '{} {} {} {} {}{}'.format(
        label, xmin, ymin, xmax, ymax, os.linesep)
    return txt


def save_groundtruth(filename, gt_path):
    file_id = os.path.basename(filename).split('.')[0]
    with open(filename, 'r') as f:
        data = f.read()
    ann = etree.fromstring(data)
    objects = ann.findall('object')
    gts = list(map(obj2txt, objects))
    out = '{}.txt'.format(file_id)
    out_path = gt_path.joinpath(out)
    with open(out_path, 'w') as f:
        f.writelines(gts)


def box2txt(box, shape, labels):
    h, w, _ = shape
    label = labels[box.get_label()]
    conf = box.get_score()
    xmin = int(box.xmin * w)
    ymin = int(box.ymin * h)
    xmax = int(box.xmax * w)
    ymax = int(box.ymax * h)
    txt = '{} {} {} {} {} {}{}'.format(
        label, conf, xmin, ymin, xmax, ymax, os.linesep)
    return txt


def save_detections(filename, image, boxes, labels, dt_path):
    file_id = os.path.basename(filename).split('.')[0]
    dts = [box2txt(box, image.shape, labels) for box in boxes]
    out = '{}.txt'.format(file_id)
    out_path = dt_path.joinpath(out)
    with open(out_path, 'w') as f:
        f.writelines(dts)


def evaluate(input_ann_path, input_img_path, output_path,
             config_path, weights_path):

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    yolo.load_weights(weights_path)

    labels = config['model']['labels']
    gt_path = output_path.joinpath('gts')
    dt_path = output_path.joinpath('dts')
    os.makedirs(gt_path, exist_ok=False)
    os.makedirs(dt_path, exist_ok=False)
    df = read_annotation(input_ann_path)
    for i, row in tqdm(df.iterrows()):
        ann_src = input_ann_path.joinpath(row.annotation)
        save_groundtruth(ann_src, gt_path)
        img_src = input_img_path.joinpath(row.image)
        image = cv2.imread(img_src.as_posix())
        boxes = yolo.predict(image)
        save_detections(img_src, image, boxes, labels, dt_path)


if __name__ == '__main__':
    main()
