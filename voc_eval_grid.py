# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import sys
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
import numpy as np
import math

def parse_size(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []

    size = tree.find('size')
    size_struct = []
    size_struct = [int(size.find("width").text), int(size.find("height").text)]
    return size_struct


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []

    size = tree.find('size')
    size_struct = []
    size_struct = [int(size.find("width").text), int(size.find("height").text)]


    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        obj_struct['size'] = size_struct
        objects.append(obj_struct)

    return objects

def get_grid_number_by_center_size(size, bbox, imagename):
    width_dict = {}
    height_dict = {}

    center_x = math.floor(bbox[0])
    center_y = math.floor(bbox[1])

    for i in range(size[0]):
        width_dict[i] =  math.floor(i / (size[0] / 7))

    for i in range(size[1]):
        height_dict[i] = math.floor(i / (size[1] / 7))

    grid = width_dict[center_x]  + height_dict[center_y] * 7

    return grid

def get_grid_number(size, bbox, imagename):
    width_dict = {}
    height_dict = {}

    center_x = math.floor((bbox[0] + bbox[2])/2)
    center_y = math.floor((bbox[1] + bbox[3])/2)

    for i in range(size[0]):
        width_dict[i] =  math.floor(i / (size[0] / 7))

    for i in range(size[1]):
        height_dict[i] = math.floor(i / (size[1] / 7))

    grid = width_dict[center_x]  + height_dict[center_y] * 7

    return grid

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results_nis3 file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_with_size.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
    # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)

    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)
    #
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        size = np.array([x['size'] for x in R])
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'size': size}

    # read dets, detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # todo, don't need sort, but I need the threshold, sort by confidence, no need to sort, instead, we need the
    # go down dets and mark TPs and FPs, 这里是全部的长度，是4952 * 147 ， 但是我要的是 4952 * 49, check for each class

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    gt = np.zeros(len(imagenames) * 49)
    for i, imagename in enumerate(imagenames):
        R = class_recs[imagename]
        BBGT = R['bbox'].astype(float)
        size = R['size']
        difficult = R['difficult']
        for k, bbgt in enumerate(BBGT):
            grid = get_grid_number(size[0], bbgt, imagename)
            if ~difficult[k]:
                gt[i*49+grid] = 1

    recs = []
    precs = []

    tpsum = []
    fpsum = []
    gtsum = []

    for index ,thresh in enumerate(np.arange(0.99, -0.01, -0.01)):
        #the confidence here can represent the detection
        det = confidence > thresh

        tp[np.where(det + gt  == 2)] = 1
        fp[np.where(det - gt == 1)] = 1
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
        recal = np.sum(tp) / np.sum(gt)
        prec = np.sum(tp) /np.maximum(np.sum(tp + fp), np.finfo(np.float64).eps)

        recs += [recal]
        precs += [prec]

        tpsum += [np.sum(tp)]
        fpsum += [np.sum(fp)]
        gtsum += [np.sum(gt)]

    return recs, precs, tpsum, fpsum, gtsum

def voc_eval_cardi(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results_nis3 file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_with_size.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)

    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)
    #
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        size = np.array([x['size'] for x in R])
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'size': size}

    # read dets, detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # todo, don't need sort, but I need the threshold, sort by confidence, no need to sort, instead, we need the
    # go down dets and mark TPs and FPs, 这里是全部的长度，是4952 * 147 ， 但是我要的是 4952 * 49, check for each class

    nd = len(image_ids)
    tp = np.zeros(len(imagenames)*49)
    fp = np.zeros(len(imagenames)*49)

    gt = np.zeros(len(imagenames) * 49)
    prid = np.zeros(len(imagenames) * 49)
    for i, imagename in enumerate(imagenames):
        R = class_recs[imagename]
        BBGT = R['bbox'].astype(float)
        size = R['size']
        difficult = R['difficult']
        for k, bbgt in enumerate(BBGT):
            grid = get_grid_number(size[0], bbgt, imagename)
            if ~difficult[k]:
                gt[i*49+grid] = 1

        for k, name in enumerate(image_ids):
            if name == imagename:
                grid = get_grid_number(parse_size(annopath.format(name)), BB.tolist()[k], name)
                prid[i*49 + grid]  = 1

    tp[np.where(prid + gt == 2)[0]] = 1
    fp[np.where(prid - gt == 1)[0]] = 1

    recs = [np.sum(tp)/np.sum(gt)]
    precs = [np.sum(tp)/(np.sum(tp) + np.sum(fp))]

    tpsum = [np.sum(tp)]
    fpsum = [np.sum(fp)]
    gtsum = [np.sum(gt)]


    return recs, precs, tpsum, fpsum, gtsum


def voc_eval_multi_label(detpath,
                   annopath,
                   imagesetfile,
                   classname,
                   cachedir,
                   ovthresh=0.5,
                   use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results_nis3 file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots_with_size.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)

    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)
    #
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        size = np.array([x['size'] for x in R])
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'size': size}

    # read dets, detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # todo, don't need sort, but I need the threshold, sort by confidence, no need to sort, instead, we need the
    # go down dets and mark TPs and FPs, 这里是全部的长度，是4952 * 147 ， 但是我要的是 4952 * 49, check for each class

    nd = len(image_ids)
    tp = np.zeros(len(imagenames))
    fp = np.zeros(len(imagenames))

    gt = np.zeros(len(imagenames))
    prid = np.zeros(len(imagenames))
    for i, imagename in enumerate(imagenames):
        R = class_recs[imagename]
        BBGT = R['bbox'].astype(float)
        size = R['size']
        difficult = R['difficult']
        for k, bbgt in enumerate(BBGT):
            grid = get_grid_number(size[0], bbgt, imagename)
            if ~difficult[k]:
                gt[i] = 1

        for k, name in enumerate(image_ids):
            if name == imagename:
                grid = get_grid_number(parse_size(annopath.format(name)), BB.tolist()[k], name)
                prid[i]  = 1

    tp[np.where(prid + gt == 2)[0]] = 1
    fp[np.where(prid - gt == 1)[0]] = 1

    recs = [np.sum(tp)/np.sum(gt)]
    precs = [np.sum(tp)/(np.sum(tp) + np.sum(fp))]

    tpsum = [np.sum(tp)]
    fpsum = [np.sum(fp)]
    gtsum = [np.sum(gt)]


    return recs, precs, tpsum, fpsum, gtsum

if __name__ == '__main__':
    grid = get_grid_number([500,272], [275.988495, 122.221786, 346.787140, 155.914566], "000067")
    print(grid)
