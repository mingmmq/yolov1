./darknet yolo valid cfg/yolo.224.cfg backup/yolo_final.weights 
python reval_voc.py results --voc_dir /home/min/data/VOCdevkit
