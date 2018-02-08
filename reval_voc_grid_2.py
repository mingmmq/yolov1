#!/usr/bin/env python

# Adapt from ->
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# <- Written by Yaping Sun

"""Reval = re-eval. Re-evaluate saved detections."""

import os, sys, argparse
import numpy as np
import cPickle

from voc_eval_grid import voc_eval

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='data/VOCdevkit', type=str)
    parser.add_argument('--year', dest='year', default='2007', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)

    parser.add_argument('--classes', dest='class_file', default='data/voc.names', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_voc_results_file_template(image_set, out_dir = 'results'):
    filename = 'comp4_det_' + image_set + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path


def show_pr_curve(precision, recall, gt_prec, gt_rec, cat):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    gt_label = "({:.2f},{:.2f})".format(gt_prec, gt_rec)
    ax.annotate(gt_label, xy=(gt_prec, gt_rec), textcoords='data')

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve vs Cardinality in YOLOv1 with 7*7 grid')
    plt.plot(gt_prec, gt_rec, marker='x', color='r')
    plt.savefig('plots/{}.png'.format(cat))

    pass

def do_python_eval(devkit_path, year, image_set, classes, output_dir = 'results'):
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    tp_sums =[]
    fp_sums = []
    nposs = []
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = get_voc_results_file_template(image_set, output_dir).format(cls)
        tp_sum, fp_sum, npos = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        # aps += [ap]

        # bit add up
        tp_sums += [tp_sum]
        fp_sums += [fp_sum]
        nposs += [npos]


        # print('AP for {} = {:.4f}'.format(cls, ap))
        # print('precs for {}: {}'.format(cls, str(prec)))
        # print('recs for {}: {}'.format(cls, str(rec)))

        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
        #     cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    # print('mean prec={},\nmean rec={}').format(mean_prec, mean_rec)

    # show_pr_curve(mean_prec, mean_rec, 0, 0, "all")

    # print('Mean AP = {:.4f}'.format(np.mean(aps)))

    stp = np.sum(tp_sums)
    sfp = np.sum(fp_sums)
    snpos = np.sum(nposs)

    recs = np.divide(tp_sums , nposs)
    precs = np.divide(tp_sums, [sum(x) for x in zip(tp_sums, fp_sums)])


    rec = stp / float(snpos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = stp / np.maximum(stp + sfp, np.finfo(np.float64).eps)
    print("prec: {}, rec:{}".format(prec, rec))

    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('-- Thanks, The Management')
    # print('--------------------------------------------------------------')



if __name__ == '__main__':
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir[0])
    with open(args.class_file, 'r') as f:
        lines = f.readlines()

    classes = [t.strip('\n') for t in lines]

    print 'Evaluating detections'
    do_python_eval(args.voc_dir, args.year, args.image_set, classes, output_dir)
