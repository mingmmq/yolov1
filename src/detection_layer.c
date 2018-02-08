#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


void print_l_information(const detection_layer l) {
    int debug = 1;
    if (!debug) {
        return;
    }

//    printf("l.softmax: %d\r\n", l.softmax);
//    printf("l.outputs %d\r\n", l.outputs);
//    printf("l.batch: %d\r\n", l.batch);    //一个batch是64张图，分成4份，就是16张图一份
//    printf("l.inputs: %d\r\n", l.inputs);  // 1715 full connect层，
//    printf("l.n %d\r\n", l.n);             //n是num 现在是3个
//    printf("l.coords: %d\r\n", l.coords);   //这个是配置里面的 coords现在是4 有4个
//    printf("l.classes: %d\r\n", l.classes); //classes的个数 现在是20
//    printf("l.class_scale: %f\r\n", l.class_scale);  //class_scale是配置的
//    printf("l.cost: %f\r\n", l.cost);        //cost 也就是loss
//    printf("l.delta size: %d\r\n", sizeof(l.delta) / sizeof(float)); //这个delta就是反向的差分存了多大,并不是2
//    printf("l.noobject_scale: %f\r\n", l.noobject_scale);
//    printf("l.output size: %d\r\n", sizeof(l.output) / sizeof(float)); //已有
}

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
//    assert(side*side*((1 + l.coords)*l.n + l.classes) + 20*21 == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    print_l_information(l);

    return l;
}





void forward_detection_layer(const detection_layer l, network_state state)
{
//    printf("Here is the forward_detection_layer\r\n");
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;




    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1,
                        l.output + index + offset);
            }
        }
    }

    if(state.train){
        float avg_iou = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));

        //用来保存各个类别的obeject有多少的
        int obj_ins_index[20] = {0};
        int max_card = 21;
        float *cardinality_truth = calloc(l.classes*max_card, sizeof(float));

        for (b = 0; b < l.batch; ++b){


            int index = b*l.inputs;            //第几个1715开始,就是一个batch中第几张图片的开始
            for (i = 0; i < locations; ++i) {           //locations = l.side * l.side
                int truth_index = (b*locations + i)*(1+l.coords+l.classes); //

                int is_obj = state.truth[truth_index]; // 有了index 有了是不是obj，然后累加，就是一共有多少 在一个图片里

                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    avg_anyobj += l.output[p_index];
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = 20;

                //没有目标的这里就计算结束了
                if (!is_obj){
                    continue;
                }

                //下面都是有目标的,这是在每个location里面的，是不是truth_index就是 truth的起点，而class_index是class的起点
                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) { //循环每个class  为什么 truth_index需要有个+1的偏移呢？
                    l.delta[class_index+j] = l.class_scale * (state.truth[truth_index+1+j] - l.output[class_index+j]);
                    *(l.cost) += l.class_scale * pow(state.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    if(state.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];

                    /* generate the cardinality truth add by minming */
                    if(state.truth[truth_index + 1 + j]) obj_ins_index[j] += 1;
                    /* end modification */

                    avg_allcat += l.output[class_index+j];
                }



                box truth = float_to_box(state.truth + truth_index + 1 + l.classes);
                truth.x /= l.side;
                truth.y /= l.side;

                //n是代表有几个bounding box,找到bounding box
                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    if(best_iou > 0 || iou > 0){
                        if(iou > best_iou){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        if(rmse < best_rmse){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(state.net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                *(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                avg_obj += l.output[p_index];
                l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);

                if(l.rescore){
                    l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                }

                l.delta[box_index+0] = l.coord_scale*(state.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+1] = l.coord_scale*(state.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+2] = l.coord_scale*(state.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+3] = l.coord_scale*(state.truth[tbox_index + 3] - l.output[box_index + 3]);
                if(l.sqrt){
                    l.delta[box_index+2] = l.coord_scale*(sqrt(state.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+3] = l.coord_scale*(sqrt(state.truth[tbox_index + 3]) - l.output[box_index + 3]);
                }

                *(l.cost) += pow(1-iou, 2);
                avg_iou += iou;
                ++count;
            }

            /* add by minming, this is the cardinality index in the output */
//            int card_index = index + locations * (l.classes + l.n * (1 + l.coords)); //1是probility，coords是坐标
////            printf("card_index %d\r\n", card_index);
//            for (i = 0; i < l.classes; ++i) {
//                int offset = i*max_card;
//                //每个class是一个独立的softmax, calculate the softmax
//                softmax(l.output + card_index + offset, max_card, 1,
//                        l.output + card_index + offset);
//                cardinality_truth[offset + obj_ins_index[i]] = 1;
//            }
//            for (int n = 0; n < l.classes; ++n) {
//                for (int k = 0; k < max_card; ++k) {
//                    float y_n = cardinality_truth[n*max_card + k];
//                    float y_o = l.output[card_index + n*max_card + k];
//                    *(l.cost) += -(y_n * log(y_o) + (1-y_n) * log(1 - y_o)) / max_card; //cross-entropy
//                    //这里的正反很重要啊，居然导致反向就反了, 但是好像这个导数也不是太对
//                    l.delta[card_index + n*max_card + k] = - l.output[card_index + n*max_card +k] + cardinality_truth[n*max_card + k];
//                }
//            }
//            printf("the cost is: %f \r\n", *(l.cost));
////            here is 49 * 20 finished, 其实可以用一个循环来打印的
//            for (int m = 0; m < l.classes; ++m) {
//                printf("[%d]%d ", m, obj_ins_index[m]);
//            }
//            printf("\r\n");
//            for (int n = 0; n < l.classes; ++n) {
//                printf("t: ");
//                for (int k = 0; k < max_card; ++k) {
//                    printf("%0.3f ", cardinality_truth[n * max_card + k]);
//                }
//                printf("\r\n");
//                printf("o: ");
//                for (int k = 0; k < max_card; ++k) {
//                    printf("%.3f ", l.output[card_index + n * max_card + k]);
//                }
//                printf("\r\n");
//            }


//            for (int n = 0; n < 20; ++n) {
//                obj_ins_index[n] = 0;
//                for (int k = 0; k < max_card; ++k) {
//                    cardinality_truth[20*n+k] = 0;
//                }
//            }
//            /* End of modification minming */
        }

        print_l_information(l);

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                    }
                }
            }
            free(costs);
        }

//        printf("cost before: %f\r\n", *(l.cost));
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
//        printf("cost after: %f\r\n", *(l.cost));


        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_detection_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / l.side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / l.side * h;
            boxes[index].w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}


void get_detection_boxes_nonms(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / l.side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / l.side * h;
            boxes[index].w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                float prob = scale*predictions[class_index+j];
                //todo 1: remove the next line to keep all the probilities, and move the next line up
                //probs[index][j] = (prob > thresh) ? prob : 0;
                //todo 2: all the probs will be saved
                probs[index][j] = prob;

            }

//            if(only_objectness){
//                probs[index][0] = scale;
//            }
        }
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network_state state)
{
    if(!state.train){
        copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
        return;
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.side*l.side*(1+l.coords+l.classes);
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_detection_layer(l, cpu_state);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_detection_layer_gpu(detection_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

