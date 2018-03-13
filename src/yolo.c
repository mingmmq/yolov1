#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "data.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

void generate_instance_distribution(char *cfgfile, char *weightfile) {
    char *train_images = "data/voc.2007.test";

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch * net.subdivisions;
    int i = *net.seen / imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    int N = plist->size;        //get the size of all the files
    char **paths = (char **) list_to_array(plist);
    float *instance_distribution = calloc(100, sizeof(float));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA_NOT_RANDOM;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    //start new thread to load all th data
//    pthread_t load_thread = load_data_in_thread(args);

    data d;
    load_args a = args;
    while (i < N) {
//        char *path = "/home/qhu/minming/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_004923.jpg";
        d = load_data_region_not_random(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);

        printf("%s\n", a.paths[0]);
        int j;
        for (j = 0; j < d.X.rows; ++j) {
            int col;
            int row;
            int instance_count = 0;
            for (row = 0; row < a.num_boxes; ++row) {
                for (col = 0; col < a.num_boxes; ++col) {
                    int index = (col+row*a.num_boxes)*(5+classes);
//                    printf("row:%d, col:%d, index:%d\n", row, col, index);
//                    int k;
//                    for (k = 0; k < a.classes + 1; ++k) {
//                        printf("\t%.2f", d.y.vals[j][index + k]);
//                    }
//                    printf("\n");
                    //index代表的每个格子，是说这个格子里面有没有object，后面还有20个class，表示是哪个类别，还有4个表示参数的
                    instance_count += d.y.vals[j][index];
                }

            }
            printf("instance_count: %d", instance_count);
            instance_distribution[instance_count] += 1;
            printf("\n");
        }

        a.paths += args.n;
        i += 1;

//        printf("Loaded: %lf seconds\n", sec(clock() - time));
        free_data(d);
    }

    int m;
    for (m = 0; m < 100; ++m) {
        printf("%2.0f, ", instance_distribution[m]);
    }
    printf("\n");
    
}

void train_yolo(char *cfgfile, char *weightfile, int backup)
{
    char *train_images = "data/voc.train";
    char *backup_directory;
    if (0 == backup) {
         backup_directory= "./backup_yolo/";
    } else if(1==backup) {
        backup_directory = "./backup_align/";
    } else if(2==backup) {
        backup_directory = "./backup_unalign/";
    }
    printf("back folder is %s\n", backup_directory);


    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

//    for (int j = 0; j < 100; ++j) {
//        printf("%s\r\n", *paths ++);
//    }

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    //start new thread to load all th data
    pthread_t load_thread = load_data_in_thread(args);

    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        //waiting for the loading thread, then continue
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);
        //在计算的时候，又用thread重新再load了, 下面的代码是跟着上一个线程，计算和load图片是并行的进行的，加速

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_yolo_detections_max(FILE **fps, char *id, box *boxes, float **probs, int grids, int n, int classes, int w, int h)
{
    int m, j;
    int k;
    float *max_prob = calloc(classes, sizeof(float));
    char **sps = calloc(classes, sizeof(char *));
    int i1;
    for (i1 = 0; i1 < classes; ++i1) {
         sps[i1] = calloc(300, sizeof(char));
    }
    for (k = 0; k < grids; ++k) {
        memset(max_prob, 0, classes*sizeof(float));
        for (m = 0; m < n; ++m) {
            int i =  k*n + m;
            float xmin = boxes[i].x - boxes[i].w / 2.;
            float xmax = boxes[i].x + boxes[i].w / 2.;
            float ymin = boxes[i].y - boxes[i].h / 2.;
            float ymax = boxes[i].y + boxes[i].h / 2.;

            if (xmin < 0) xmin = 0;
            if (ymin < 0) ymin = 0;
            if (xmax > w) xmax = w;
            if (ymax > h) ymax = h;

            for (j = 0; j < classes; ++j) {
                if(m == 0) {
                    sprintf(sps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                            xmin, ymin, xmax, ymax);
                }
                //这是由于pythond的括号不写的缘故, 导致下面的坐标都没有正确的更新，虽然max后面改了更新了
                if (probs[i][j] >= max_prob[j]) {
                    max_prob[j] = probs[i][j];
                    sprintf(sps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                            xmin, ymin, xmax, ymax);
                }
            }
        }

        int l;
        for (l = 0; l < classes; ++l) {
            fprintf(fps[l], sps[l]);
        }
    }

    free(max_prob);
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}


void validate_yolo_cardi_print(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results_cardi/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    float *cardinalities = calloc(l.classes, sizeof(float));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
//            get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            //这是带着了图片本身的w 和 h了
            get_detection_boxes_and_cardinality_unalign(l, w, h, thresh, probs, boxes, 0, cardinalities);
//            if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            int k;
//            for (k = 0; k < l.classes; ++k) {
//                print_probs(probs, l.side * l.side*l.n, k, "test");
//            }
            do_cardi_filter(boxes, probs, l.side * l.side , l.n,  classes, cardinalities);
            print_yolo_detections(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_grid_all(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results_grid/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            //todo 1 keep all the predictions, nothing changed the negative ones are removed
            get_detection_boxes_nonms(l, w, h, thresh, probs, boxes, 0);
            //todo 2 remove the nms part, instead we are using cardinality
//            if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            //todo 3: save the result file not change
            print_yolo_detections_max(fps, id, boxes, probs, l.side*l.side, l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_cardi_gt(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results_gt/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    float *gt_cardinalities = calloc(l.classes, sizeof(float));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        //那两个参数有什么作用呢, 这是不同的坐标，只要不是
        int w = orig.w;
        int h = orig.h;
        printf("w %d h %d", w, h);
        get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
//        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        memset(gt_cardinalities, 0, l.classes * sizeof(float));
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (j = 0; j < num_labels; ++j) {
            gt_cardinalities[truth[j].id] += 1;
        }
        //filter the probs according to the cardinality
        do_cardi_filter(boxes, probs, l.side*l.side, l.n, l.classes, gt_cardinalities);
        print_yolo_detections(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h);

        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        //那两个参数有什么作用呢
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }

            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void print_cardi_dist(float **cardi_dist, int classes, char *head) {
    int cardi_distribution[20] = {17, 7, 28, 15, 21, 7, 13, 6, 21, 13, 10, 9, 8, 10, 18, 20, 22, 6, 6, 33};
    printf(head);
    printf("\n");
    int i;
    for (i = 0; i < classes; ++i) {
        printf("%s, \t", voc_names[i]);
        int j;
        for (j = 0; j < cardi_distribution[i]; ++j) {
            printf("%3d, ", (int)cardi_dist[i][j]);
        }
        printf("\n");
    }
}

void print_cardi_dist_float(float **cardi_dist, int classes, char *head) {
    int cardi_distribution[20] = {17, 7, 28, 15, 21, 7, 13, 6, 21, 13, 10, 9, 8, 10, 18, 20, 22, 6, 6, 33};
    printf(head);
    printf("\n");
    int i;
    for (i = 0; i < classes; ++i) {
        int j;
        printf("%s, \t", voc_names[i]);
        for (j = 0; j < cardi_distribution[i]; ++j) {
            printf("%.4f, ", cardi_dist[i][j]);
        }
        printf("\n");
    }
}


void validate_yolo_cardi_verify(char *cfgfile, char *weightfile)
{
    int cardi_distribution[20] = {17, 7, 28, 15, 21, 7, 13, 6, 21, 13, 10, 9, 8, 10, 18, 20, 22, 6, 6, 33};
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    float *cardinalities = calloc(l.classes, sizeof(float));
    float **cardi_all = calloc(plist->size, sizeof(float *));
    int l2;
    for (l2 = 0; l2 < plist->size; ++l2) cardi_all[l2] = calloc(classes, sizeof(float));
    float *truth_cardi = calloc(l.classes, sizeof(float));

    float **truth_all = calloc(plist->size, sizeof(float *));
    for (l2 = 0; l2 < plist->size; ++l2) truth_all[l2] = calloc(classes, sizeof(float));

    float *errors = calloc(l.classes, sizeof(float));
    float *mean_errors = calloc(l.classes, sizeof(float));
    float *gt_cardi_sum = calloc(l.classes, sizeof(float));
    float *pr_cardi_sum = calloc(l.classes, sizeof(float));


    float **gt_cardi_distribution = calloc(l.classes, sizeof(float *));
    float **pr_cardi_distribution = calloc(l.classes, sizeof(float *));
    float **er_cardi_distribution = calloc(l.classes, sizeof(float *));
    float **mean_er_cardi_dist = calloc(l.classes, sizeof(float *));
    float **std_dev_cardi_dist = calloc(l.classes, sizeof(float *));

    int j3;
    for (j3 = 0; j3 < l.classes; ++j3) {
        gt_cardi_distribution[j3] = calloc(cardi_distribution[j3], sizeof(float));
        pr_cardi_distribution[j3] = calloc(cardi_distribution[j3], sizeof(float));
        er_cardi_distribution[j3] = calloc(cardi_distribution[j3], sizeof(float));
        mean_er_cardi_dist[j3] = calloc(cardi_distribution[j3], sizeof(float));
        std_dev_cardi_dist[j3] = calloc(cardi_distribution[j3], sizeof(float));
    }



    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int total_predict = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    float error_total = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
//        get_detection_boxes(l, orig.w, orig.h, thresh, probs, boxes, 1);
        get_detection_boxes_and_cardinality_unalign(l, 1, 1, 0, probs, boxes, 0, cardinalities);
//        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        total += num_labels;

        //现在获取proposal的方式不一样，所以现在的proposal的个数就是cardinality的个数
        int n;
        int cur_labels = 0;
        for (n = 0; n < l.classes; ++n) {
            proposals += cardinalities[n];
            cur_labels += cardinalities[n];
        }

        printf(path);
        printf(" gt %d", num_labels);
        printf(" pr %d\n", cur_labels);

        //这里是计算预测的结果是不是符合iou的thresh要求,现在没有thresh了
        int m1;
        for (m1 = 0; m1 < l.classes; ++m1) {
            truth_cardi[m1] = 0;
        }
        for (j = 0; j < num_labels; ++j) {
            truth_cardi[truth[j].id] += 1;
        }


        //collect the cardinality information
        printf("i is %d\n", i);
        int j2;
        for (j2 = 0; j2 < l.classes; ++j2) {
            cardi_all[i][j2] = cardinalities[j2];
            truth_all[i][j2] = truth_cardi[j2];

            int gt_cardi = truth_cardi[j2];
            int pr_cardi = cardinalities[j2];
            gt_cardi_distribution[j2][gt_cardi] += 1;
            pr_cardi_distribution[j2][pr_cardi] += 1;

            int i1;
            for (i1 = 0; i1 < cardi_distribution[j2]; ++i1) {
                if ((gt_cardi == i1 && pr_cardi != i1) || (gt_cardi != i1 && pr_cardi == i1)) {
                    er_cardi_distribution[j2][i1] += 1;
                }
                mean_er_cardi_dist[j2][i1] = er_cardi_distribution[j2][i1] / (i+1);
            }
        }

        print_cardi_dist(gt_cardi_distribution, l.classes, "gt");
        print_cardi_dist(pr_cardi_distribution, l.classes, "prediction");
        print_cardi_dist(er_cardi_distribution, l.classes, "error");
        print_cardi_dist_float(mean_er_cardi_dist, l.classes, "mean");

        printf("truth:");
        int l1;
        for (l1 = 0; l1 < l.classes; ++l1) {
            int i1;
            gt_cardi_sum[l1] += truth_cardi[l1];
            for (i1 = 0; i1 < truth_cardi[l1]; ++i1) {
                printf(" %d", l1);
            }
        }
        printf("\n");


        printf("predi");
        int i1;
        for (i1 = 0; i1 < l.classes; ++i1) {
            int k1;
            pr_cardi_sum[i1] += cardinalities[i1];
            for (k1 = 0; k1 < cardinalities[i1]; ++k1) {
                total_predict ++;
                printf(" %d", i1);
            }
        }
        printf("\n");

        printf("error, ");
        int n1;
        for (n1 = 0; n1 < l.classes; ++n1) {
            errors[n1] += abs(cardinalities[n1] - truth_cardi[n1]);
            mean_errors[n1] = errors[n1] / (i + 1);
            printf("%f, ", errors[n1]);
        }
        printf("\n");

        printf("true_cardi_sum, ");
        int k2;
        for (k2 = 0; k2 < l.classes; ++k2) {
            printf("%f, ", gt_cardi_sum[k2]);
        }
        printf("\n");


        printf("predict_cardi_sum, ");
        int m2;
        for (m2 = 0; m2 < l.classes; ++m2) {
            printf("%f, ", pr_cardi_sum[m2]);
        }
        printf("\n");



        printf("mean_errors, ");
        int i2;
        for (i2 = 0; i2 < l.classes; ++i2) {
            printf("%f, ", mean_errors[i2]);
        }
        printf("\n");

        printf("total %d ", total);
        printf("total pridict %d ", total_predict);

//        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }

    for (i = 0; i < m; ++i) {
        int n;
        for (n = 0; n < l.classes; ++n) {
            int i1;
            for (i1 = 0; i1 < cardi_distribution[n]; ++i1) {
                if (cardi_all[i][n] != truth_all[i][n] && (cardi_all[i][n] == i1 || truth_all[i][n] == i1) ) {
                    std_dev_cardi_dist[n][i1] += pow((1 - mean_er_cardi_dist[n][i1]), 2);
                } else {
                    std_dev_cardi_dist[n][i1] += pow((0 - mean_er_cardi_dist[n][i1]), 2);
                }
            }
        }
    }
    int n;
    for (n = 0; n < l.classes; ++n) {
        int i1;
        for (i1 = 0; i1 < cardi_distribution[n]; ++i1) {
            std_dev_cardi_dist[n][i1] = pow(std_dev_cardi_dist[n][i1]/(m-1), 0.5);
        }
    }
    print_cardi_dist_float(std_dev_cardi_dist, l.classes, "std_dev");


    int i2;
    for (i2 = 0; i2 < l.classes; ++i2) {
        error_total += errors[i2];
    }
    printf("error sum is %f ", error_total);
    printf("error mean is %f\n", error_total / (m * l.classes));

    int k2;
    //use errors as the temporary parameter
    memset(errors, 0, l.classes* sizeof(float));
    for (k2 = 0; k2 < m; ++k2) {
        int n;
        for (n = 0; n < l.classes; ++n) {
            errors[n] += pow(cardi_all[k2][n] - mean_errors[n], 2);
        }
    }
    int m2;
    printf("std_dev, ");
    for (m2 = 0; m2 < l.classes; ++m2) {
        float std_dev = (float) pow(errors[m2] / (m - 1), 0.5);
        printf(" %f, ", std_dev);
    }
    printf("\n");



    int n2;
    float error_overall;
    for (n2 = 0; n2 < m; ++n2) {
        int n;
        for (n = 0; n < l.classes; ++n) {
            float cur_error = abs(cardi_all[n2][n] - truth_all[n2][n]);
            error_overall += cur_error;
//            printf("[%d][%d] %f \n", n2, n, cur_error);
        }
    }
    printf(" error_overall %f", error_overall);
    float error_mean_overall = error_overall / (m * l.classes);
    printf(" error mean overall %f", error_mean_overall);

    int i3;
    float cal_mean;
    for (i3 = 0; i3 < l.classes; ++i3) {
        cal_mean += mean_errors[i3];
    }
    printf(" cal error mean overall %f ", cal_mean/l.classes);

    float std_dev_tmp = 0;
    for (k2 = 0; k2 < m; ++k2) {
        int n;
        for (n = 0; n < l.classes; ++n) {
             std_dev_tmp += pow(abs(cardi_all[k2][n]-truth_all[k2][n]) - error_mean_overall, 2);
        }
    }
    printf(" std dev temp %f", std_dev_tmp);
    float std_dev_overall = pow(std_dev_tmp / (m - 1), 0.5);
    printf(" std dev overall %f", std_dev_overall);

    printf("\n");
}


void validate_yolo_cardi_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results_cardi/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float));

    float *cardinalities = calloc(l.classes, sizeof(float));
    float **cardi_all = calloc(plist->size, sizeof(float *));
    int l2;
    for (l2 = 0; l2 < plist->size; ++l2) cardi_all[l2] = calloc(classes, sizeof(float));
    float *truth_cardi = calloc(l.classes, sizeof(float));
    float **truth_all = calloc(plist->size, sizeof(float *));
    for (l2 = 0; l2 < plist->size; ++l2) truth_all[l2] = calloc(classes, sizeof(float));
    float *errors = calloc(l.classes, sizeof(float));
    float *mean_errors = calloc(l.classes, sizeof(float));
    float *gt_cardi_sum = calloc(l.classes, sizeof(float));
    float *pr_cardi_sum = calloc(l.classes, sizeof(float));


    int m = plist->size;
    int i=0;

    float thresh = .000;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int total_predict = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
    float error_total = 0;

    for(i = 0; i < m; ++i){

        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
//        get_detection_boxes(l, orig.w, orig.h, thresh, probs, boxes, 1);

        get_detection_boxes_and_cardinality_unalign(l, 1, 1, 0, probs, boxes, 0, cardinalities);
//        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
//        total += num_labels;
        printf("num labels %d\n", num_labels);
        //现在获取proposal的方式不一样，所以现在的proposal的个数就是cardinality的个数
        int n;
        for (n = 0; n < l.classes; ++n) {
            proposals += cardinalities[n];
        }

        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
//                printf("prob %f iou %f \n", probs[k][0], iou);
                if(iou > best_iou){
                    best_iou = iou;
                }
            }
//            printf("%s best iou %f \n", path, best_iou);
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }

}


void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}



void test_yolo_cardi(char *cfgfile, char *weightfile, char *filename, float thresh, int align)
{
    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    float *cardinalities = calloc(l.classes, sizeof(float));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
//        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (align) {
            get_detection_boxes_and_cardinality_align(l, 1, 1, thresh, probs, boxes, 0, cardinalities);
        } else {
            get_detection_boxes_and_cardinality_unalign(l, 1, 1, 0, probs, boxes, 0, cardinalities);
        }
//        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
        draw_detections_by_cardi(im, l.side * l.side , l.n, thresh, boxes, probs, voc_names, alphabet, 20,
                                 cardinalities);
        int k;
        for (k = 0; k < l.classes; ++k) {
            printf("[%d] cardinality", k);
            int l;
            printf(" %f", cardinalities[k]);
            printf("\n");
        }
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void print_yolo_detections_and_cardi(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h,
                                     float *cardinalities)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                                     xmin, ymin, xmax, ymax);
        }
    }

    int k;
    for (k = 0; k < classes; ++k) {
        fprintf(fps[k], "cardinality");
        fprintf(fps[k], " %f", cardinalities[k]);
        fprintf(fps[k], "\n");
    }

}


void validate_yolo_print_cardi(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results_cardi/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    int *cardinalities = calloc(21*l.classes, sizeof(float));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;

            fprintf(stderr, "l.outpus %d \n", l.outputs);

            //todo 1 keep all the predictions
            get_detection_boxes_and_cardinality_align(l, w, h, thresh, probs, boxes, 0, cardinalities);
            //todo 2 remove the nms part, instead we are using cardinality
            //if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            //todo 3: save the result file not change
            print_yolo_detections_and_cardi(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h, cardinalities);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void run_yolo(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    if(0==strcmp(argv[2], "test_ca")) test_yolo_cardi(cfg, weights, filename, thresh, 1);
    if(0==strcmp(argv[2], "test_cu")) test_yolo_cardi(cfg, weights, filename, thresh, 0);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights, 0);
    else if(0==strcmp(argv[2], "train_ca")) train_yolo(cfg, weights, 1);
    else if(0==strcmp(argv[2], "train_cu")) train_yolo(cfg, weights, 2);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid_cardi")) validate_yolo_cardi_print(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "recall_cardi")) validate_yolo_cardi_recall(cfg, weights);
    else if(0==strcmp(argv[2], "valid_grid")) validate_yolo_grid_all(cfg, weights);
    else if(0==strcmp(argv[2], "cardi")) validate_yolo_print_cardi(cfg, weights);
    else if(0==strcmp(argv[2], "valid_gt")) validate_yolo_cardi_gt(cfg, weights);
    else if(0==strcmp(argv[2], "verify")) validate_yolo_cardi_verify(cfg, weights);
    else if(0==strcmp(argv[2], "distribution")) generate_instance_distribution(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix);
}
