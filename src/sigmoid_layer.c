//
// Created by Qian Minming on 6/3/18.
//
#include "sigmoid_layer.h"
#include "blas.h"
#include "cuda.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

sigmoid_layer make_sigmoid_layer(int batch, int inputs){
    fprintf(stderr, "sigmoid                                        %4d\n",  inputs);
    sigmoid_layer l = {0};
    l.type = SIGMOID;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs * batch, sizeof(float));
    l.delta = calloc(inputs * batch, sizeof(float));

    l.forward = forward_sigmoid_layer;
    l.backward = backward_sigmoid_layer;

#ifdef GPU
    l.forward_gpu = forward_sigmoid_layer_gpu;
    l.backward_gpu = backward_sigmoid_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif

    return l;
}

void forward_sigmoid_layer(const sigmoid_layer l, network_state state){
    int b;
    int inputs = l.inputs;
    int batch = l.batch;
    for (b = 0; b < batch; ++b) {
        sigmoid(state.input + b * inputs, inputs, l.output + b * inputs);
    }
}

void backward_sigmoid_layer(const sigmoid_layer l, network_state state) {
    int i;
    for (i = 0; i < l.inputs*l.batch; ++i) {
        state.delta[i] += l.delta[i];
    }
}

#ifdef GPU
void forward_sigmoid_layer_gpu(const sigmoid_layer l, network_state state) {
    int inputs = l.inputs;
    int batch = l.batch;
    sigmoid_gpu(state.input, inputs, inputs, batch, l.output_gpu);
}

void backward_sigmoid_layer_gpu(const sigmoid_layer layer, network_state state){
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}
#endif