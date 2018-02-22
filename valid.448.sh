./darknet yolo valid cfg/yolo.test.448.cfg yolov1.448.weights
python reval_voc.py results --voc_dir ~/Github/data/pascal/VOCdevkit
