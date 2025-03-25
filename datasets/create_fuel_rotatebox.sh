#!/bin/bash
# 这个是给定旋转盒子的方向，重力不变
fuyang=40 #俯仰角
henggun=10 #横滚角
seed=999
name=${fuyang}f${henggun}h
frame=0800

OUTPUT_SCENES_DIR=scenes/fuel_triangletank/noban/$name
OUTPUT_DATA_DIR=scenes/fuel_triangletank/zst/noban/$name
OUTPUT_NPZ_DIR=scenes/test/noban/$name
OUTPUT_NPZ_FILE=scenes/test/noban/$name/fluid_${frame}.npz
OUTPUT_BGEO_FILE=fluid_${frame}.bgeo
OUTPUT_OBJ_FILE=surface_e${frame}.obj

mkdir $OUTPUT_SCENES_DIR

# generate raw data
python datasets/create_physics_scenes_fuel_triangletank_rotatebox.py --output $OUTPUT_SCENES_DIR \
                            --seed $seed \
                            --rotated_obj_path=/home/cyu/code/dual_small/scenes/fuel_triangletank/rotated_obj/noban \
                            --fuyang $fuyang \
                            --henggun $henggun

# raw to zst
python datasets/create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR

# zst to npz
python scripts/run_network_fueltank.py --weights 52001_model_weights_best.pt \
                                        --scene scripts/example_scene.json \
                                        --zst_dir $OUTPUT_DATA_DIR \
                                        --output $OUTPUT_NPZ_DIR \
                                        --num_steps 400
# npz to bgeo
python npz2bgeo.py --path $OUTPUT_NPZ_FILE

# bgeo to obj
cd $OUTPUT_NPZ_DIR
splashsurf reconstruct $OUTPUT_BGEO_FILE -r=0.023 -l=1.8 -c=0.5 -t=0.6 --subdomain-grid=on --mesh-cleanup=on \
            --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10 -o $OUTPUT_OBJ_FILE

##Split data in train and validation set
#mkdir $OUTPUT_DATA_DIR/train
#mkdir $OUTPUT_DATA_DIR/valid
#
#for seed in `seq -f "%04g" 1 62`; do
#        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/train
#done
#
#for seed in `seq -f "%04g" 73 80`; do
#        mv $OUTPUT_DATA_DIR/sim_${seed}_*.msgpack.zst $OUTPUT_DATA_DIR/valid
#done
