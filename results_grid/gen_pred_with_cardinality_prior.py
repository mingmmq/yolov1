import os
import operator
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer

def get_filename(img_path, data):
    return os.path.join(img_path, "{}.jpg".format(str(data[0]).zfill(6)))

def get_annofile(anno_path, data):
    return os.path.join(anno_path, "{}.xml".format(str(data[0]).zfill(6)))

# annotation operations
def load_annotation(anno_filename):
    with open(anno_filename) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, "lxml")

#get the label from the the cordinates provided by the result
#should check if the cordinates are the same as the yolov1
def getLabel(anno_path, data, img_path, cat=0):
    grid = 7

    img_name = "{}.jpg".format(str(data[0]).zfill(6))

    anno_file = get_annofile(anno_path, data)
    anno = load_annotation(anno_file)
    objs = anno.findAll('object')
    size = anno.find('size')
    width = float(size.findChild('width').contents[0])
    height = float(size.findChild('height').contents[0])

    grid_width = width/grid
    grid_height = height/grid

    cx = (data[4]-data[2]) / 2 + data[2]
    gx = int(cx/grid_width)
    cy = (data[5]-data[3]) / 2 + data[3]
    gy = int(cy/grid_height)

    pred_position = gy * grid + gx
    # print(anno_file, pred_position)


    #this line is for visual test only
    # utils.draw_detection("/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/", img_name, data)

    return pred_position + grid*grid*cat

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
    for tag in df['tags'].str.split().tolist():
        if type(tag) is float:
            labels.append([])
        else:
            labels.append([int(one_tag) for one_tag in  tag])

    return img_names, labels



def generate_prediction(count_data, file_names, grid):
    anno_path = "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/Annotations"
    img_path = "/Users/qianminming/Github/data/pascal/VOCdevkit/VOC2007/JPEGImages"
    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    #covner the integer to string, adding back the leading zeros and return the result as the string.
    #str(1).zfill(6)


    result_table = {}

    # go through each category of the objects
    for cate in pascal_class_dic:
        result_file_name = "comp4_det_test_{}.txt".format(cate)
        data = np.genfromtxt(result_file_name, dtype=[int, float, float, float, float, float],delimiter=" ")

        # this is the data for one category, so we have to find which file include the instances
        # we use the groundtruth information  and then only detect those index are the same with each images
        #


        #I should ge 98 number of the files and only keep half of that
        file_count = -1
        # go though each line of cardinality prior file
        for line, filename in zip(count_data, file_names):
            #only select those above the thresh
            # use the file name as result table is correct, I finally need to store them in the result table
            #the file name part can be used the index for indexing which images and show the matrix

            #in the ground truth, if this file not include the item in this category, then just skipp
            file_count += 1
            # k is the value of the column for different category
            k = line[pascal_class_dic[cate]]

            # ignore the those with no objects of this category in this image
            if k == 0:
                continue

            #todo here need to modify the box number, some times it is about 147 numbers
            box_numbers = grid*grid*3
            this_file_score_list = operator.itemgetter(range(file_count*box_numbers,(file_count+1)*box_numbers))(data)
            only_socore_list = [x[1] for x in this_file_score_list]
            #this line will only keep the first, but we need to keem the max one
            first_value_array = only_socore_list[0::3]
            second_value_array = only_socore_list[1::3]
            third_value_array = only_socore_list[2::3]

            keep_max = []
            for first, second, third in zip(first_value_array, second_value_array, third_value_array):
                keep_max.append(max(max(first, second),third))

            np_scores = np.array(keep_max)


            #find the top k in the file, here is how we find the top k
            #and the gt_local_grids are the grid number
            gt_local_grids = np_scores.argsort()[-k:][::-1]

            highest_scores = []
            for i in gt_local_grids:
                highest_scores.append(keep_max[i])

            highest_scores_rows = []
            for l in this_file_score_list:
                if l[1] in highest_scores:
                    highest_scores_rows.append(l)

            global_grids = []
            for row in highest_scores_rows:
                g_grid = getLabel(anno_path, row, img_path, pascal_class_dic.get(cate))
                global_grids.append(g_grid)

            # the file name will store all the global of grids in the image
            # then concat them into the list stored in the dictionary
            # global_grids = [x+pascal_class_dic[cate]*grid*grid for x in g_grid_list]

            #have the instance, then we can get the top k items
            if filename not in result_table:
                result_table[filename]  = []

            # done
            # no need for the thresh, now we need to get the top k numbers, for each grid,
            # the k numbers are stored in the dataset in the test file
            # if item[1] > thresh:
            # todo: get the target by the specific number of items
            # the cound is stored by files, so we need to iterate one by one by files, but for each file
            # is there all the image have the predicion for each grid,
            # 1. file only have the dog and the person, this



            result_table[filename] += global_grids
            print(filename, result_table[filename])

    img_names = []
    labels = []
    keylist = result_table.keys()
    keylist.sort()

    for key in keylist:
        print(key, result_table[key])
        img_names.append("{}.jpg".format(key))
        labels.append(result_table[key])

    yolo_grid_csv = "yolo_grid_ground_truth_max.csv".format()
    toCsv(yolo_grid_csv, img_names, labels)



if __name__ == '__main__':


    csv_file = "test_grid_voc2007_grid_7_count.csv"
    pred_img_names, counts = fromCsv(csv_file)

    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    # generate_prediction()

    num_to_name_dict = dict((y,x) for x,y in pascal_class_dic.iteritems())

    #one by one generate the grid and save them into the csv file
    grid = 7

    #then read from the csv file, then the problem solved, generate the acc, prec and the f1 score
    generate_prediction(counts, pred_img_names, grid)

    pass
