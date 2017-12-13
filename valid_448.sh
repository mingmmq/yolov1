./darknet yolo valid cfg/yolo.cfg yolov1_448.weights 
python reval_voc.py results --voc_dir /home/min/data/VOCdevkit
