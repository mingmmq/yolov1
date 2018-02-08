import numpy as np
import pandas as pd
import operator

def generate_result(cardinality_file, anchors):
    '''
    1. read the result of different comp4 result
    2. the thresh is used to select the rows above the thresh
    :param anchors:
    :return: nothing
    '''

    #read file
    #todo here i need to get the result from the cardinality prior

    filter_row_by_cardinality_prior(cardinality_file, anchors)



def fromcsv(csv_file):
    df = pd.read_csv(csv_file)
    img_names = df['img']
    labels = []
    for tag in df['tags'].str.split().tolist():
        if type(tag) is float:
            labels.append([])
        else:
            labels.append([int(one_tag) for one_tag in  tag])

    return img_names, labels


def get_max(compare_column, list_of_one_grid):
    # this is an arbitrary value to select at least row
    max = -100
    the_row = ""
    for row in list_of_one_grid:
        if row[compare_column] > max:
            max = row[compare_column]
            the_row = row

    return the_row


def filter_row_by_cardinality_prior(cardinality_file, anchors):
    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    pred_img_names, counts = fromcsv(cardinality_file)

    # go through each category of the objects
    for cate in pascal_class_dic:
        filtered_list = []
        result_file_name = "comp4_det_test_{}.txt".format(cate)
        #we can read data
        data = np.genfromtxt(result_file_name, dtype=[int, float, float, float, float, float],delimiter=" ")

        # this is the data for one category, so we have to find which file include the instances
        # we use the groundtruth information  and then only detect those index are the same with each images
        #


        #I should ge 98 number of the files and only keep half of that
        file_count = -1
        # go though each line of cardinality prior file
        for line, filename in zip(counts, pred_img_names):
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
            box_numbers = grid*grid*anchors
            this_file_score_list = operator.itemgetter(range(file_count*box_numbers,(file_count+1)*box_numbers))(data)

            best_rows =[]
            for i in range(0, grid*grid):
                anchor_rows = operator.itemgetter(range(i*anchors, (i+1)*anchors))(this_file_score_list)
                best_row = get_max(1, anchor_rows)
                best_rows.append(best_row)
                pass


            # the list of the value of the second column and then get index of the top k(in the count)
            only_socore_list = [x[1] for x in best_rows]
            indexes = np.array(only_socore_list).argsort()[-k:][::-1]
            predict_list = [best_rows[index] for index in indexes]

            filtered_list = filtered_list + predict_list


            ## save the file here, no need to return
            # this part is convert the row_list to the output file
        folder_name = "result_{}".format("cardinality_prior")
        with open("{}/{}".format(folder_name,result_file_name), 'w') as fp:
            output_string = ""
            for row in filtered_list:
                name = "{}".format(row[0]).zfill(6)
                temp_str = " {} {} {} {} {}\n".format(row[1], row[2], row[3], row[4], row[5])
                output_string += name + temp_str

            fp.write(output_string)
        pass

    # return filtered_list


def create_folder_by_thresh(folder_name):
    import os
    try:
        os.mkdir(folder_name)
    except:
        pass


def generate_all_cardinality_prior_predicitons(cardinality_file, anchors):
    pascal_class_dic = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


    folder_name = "result_{}".format("cardinality_prior")
    create_folder_by_thresh(folder_name)
    # for category in pascal_class_dic.keys():
    #     input_result_file = "comp4_det_test_{}.txt".format(category)
    #     output_file_path = "{}/comp4_det_test_{}.txt".format(folder_name, category)

    generate_result(cardinality_file, anchors)


if __name__ == '__main__':
    grid = 7
    anchors = 3
    cardinality_file = "test_grid_voc2007_grid_{}_count.csv".format(grid)
    generate_all_cardinality_prior_predicitons(cardinality_file, anchors)


