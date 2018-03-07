//
// Created by Qian Minming on 6/3/18.
//

#ifndef YOLOV1_SIGMOID_LAYER_H
#define YOLOV1_SIGMOID_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer sigmoid_layer;

sigmoid_layer make_sigmoid_layer(int batch, int inputs);
void forward_sigmoid_layer(const sigmoid_layer l, network_state state);
void backward_sigmoid_layer(const sigmoid_layer l, network_state state);

#ifdef GPU
void forward_sigmoid_layer_gpu(const sigmoid_layer l, network_state state);
void backward_sigmoid_layer_gpu(const sigmoid_layer l, network_state state);
#endif

#endif //YOLOV1_SIGMOID_LAYER_H

