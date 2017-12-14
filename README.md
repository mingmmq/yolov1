![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


# yolov1
This version is checkout from 

```git checkout c71bff69eaf1e458850ab78a32db8aa25fee17dc```

More informaiton, please refer to [This issue](https://github.com/pjreddie/darknet/issues/99) 

##How to train yolov1
The extraction.conv.weights can be fetched from:

```wget http://pjreddie.com/media/files/extraction.conv.weights```


To train the 448*448 resolution version:

```./darknet yolo train cfg/yolo.train.cfg extraction.conv.weights```


To train the 224*224 resolution version:

```./darknet yolo train cfg/yolo.train.224.cfg extraction.conv.weights```


##To validate the mAP metrics of the project 
run `valid.448.sh`

it will call the python script provide by [This Repository](https://github.com/muchuanyun/darknet)

