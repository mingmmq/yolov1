import numpy as np
import os
from bs4 import BeautifulSoup
import utils
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def get_filename(img_path, data):
    return os.path.join(img_path, "{}.jpg".format(str(data[0]).zfill(6)))

def get_annofile(anno_path, data):
    return os.path.join(anno_path, "{}.xml".format(str(data[0]).zfill(6)))

def getLabel(anno_path, data, img_path, cat=0):
    img_name = "{}.jpg".format(str(data[0]).zfill(6))

    anno_file = get_annofile(anno_path, data)
    anno = load_annotation(anno_file)
    objs = anno.findAll('object')
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    grid_width = width/7
    grid_height = height/7

    cx = (data[4]-data[2]) / 2 + data[2]
    gx = int(cx/grid_width)
    cy = (data[5]-data[3]) / 2 + data[3]
    gy = int(cy/grid_height)

    pred_position = gy * 7 + gx
    # print(anno_file, pred_position)


    #this line is for visual test only
    # utils.draw_detection("/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/", img_name, data)

    return pred_position + 7*7*cat

# annotation operations
def load_annotation(anno_filename):
    with open(anno_filename) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "lxml")

def toCsv(csv_file, img_names, labels):
    tags_string = []
    for tags in labels:
        tag_string = " ".join([str(tag) for tag in tags])
        tags_string.append(tag_string)

    df = pd.DataFrame({'img':img_names, 'tags': tags_string})
    df.to_csv(csv_file)

def fromCsv(csv_file):
    df = pd.read_csv(csv_file)
    img_names = df['img']
    labels = []
    import math
    for tag in df['tags'].str.split().tolist():
        if type(tag) is float:
            labels.append([])
        else:
            labels.append([int(one_tag) for one_tag in  tag])

    return img_names, labels

def generate_prediction(thresh):
    anno_path = "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/Annotations"
    img_path = "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/JPEGImages"
    s = "comp4_det_test_person.txt"
    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    #covner the integer to string, adding back the leading zeros and return the result as the string.
    #str(1).zfill(6)


    result_table = {}

    for cate in pascal_class_dic:
        result_file_name = "comp4_det_test_{}.txt".format(cate)
        data = np.genfromtxt(result_file_name, dtype=[int, float, float, float, float, float],delimiter=" ")

        for item in data:
            #only select those above the thresh
            filename = str(int(item[0])).zfill(6)
            if filename not in result_table:
                result_table[filename]  = []
            if item[1] > thresh:
                result_table[filename].append(getLabel(anno_path, item, img_path, pascal_class_dic.get(cate)))

    img_names = []
    labels = []
    keylist = result_table.keys()
    keylist.sort()

    for key in keylist:
        # print(key, result_table[key])
        img_names.append("{}.jpg".format(key))
        labels.append(result_table[key])

    yolo_grid_csv = "yolo_grid_thresh{}.csv".format(thresh)
    toCsv(yolo_grid_csv, img_names, labels)


def generateResult(pred_file, target_file):
    classes = 20
    grids = 7
    pred_csv_file = pred_file
    pred_img_names, labels = fromCsv(pred_csv_file)
    mlb = MultiLabelBinarizer(range(classes*grids*grids))
    pred = mlb.fit_transform(labels).astype(np.float32)

    gt_image_names, gt_labels = fromCsv(target_file)
    target = mlb.fit_transform(gt_labels).astype(np.float32)

    tp = np.sum((pred + target) == 2)
    fp = np.sum((pred - target) == 1)
    fn = np.sum((pred - target) == -1)
    tn = np.sum((pred + target) == 0)
    acc = (tp + tn + 0.0) / (tp + tn + fp + fn)
    prec = (tp + 0.0) / (tp + fp)
    rec = (tp + 0.0) / (tp + fn)
    f1 = 2.0*(prec*rec)/(prec+rec)

    print("acc:{}, prec:{}, rec:{}, f1:{}".format(acc, prec, rec, f1))

if __name__ == '__main__':

    import sys
    thresh = float(sys.argv[1])


    target_file =  "test_grid_voc2007_grid_7.csv"
    pred_file = "yolo_grid_ground_truth_max.csv"

    # todo comment this in and out to run the thresh detection on the nms data
    for thresh in np.arange(0.1, 1, 0.1):
        generate_prediction(thresh)
        print("thresh: {}".format(thresh))
        pred_file = "yolo_grid_thresh{}.csv".format(thresh)

        generateResult(pred_file, target_file)

    pass
