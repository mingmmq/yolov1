#!/usr/bin/env bash
for i in {1..9}
do
    echo python reval_voc.py results_grid/result_thresh_0.$i --voc_dir /Users/qianminming/Github/data/pascal/VOCdevkit
    python reval_voc.py results_grid/result_thresh_0.$i --voc_dir /Users/qianminming/Github/data/pascal/VOCdevkit
done

echo python reval_voc.py results_grid/result_cardinality_prior --voc_dir /Users/qianminming/Github/data/pascal/VOCdevkit
python reval_voc.py results_grid/result_cardinality_prior --voc_dir /Users/qianminming/Github/data/pascal/VOCdevkit

