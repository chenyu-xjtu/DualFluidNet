#!/bin/bash
# 这个是给定旋转重力方向，盒子不动
tank_name=69R_tank
fuyang=0 #俯仰角
henggun=0 #横滚角
version=last_image_0_to10
name=${fuyang}f${henggun}h_$version
#name=train
frame=0299

OUTPUT_SCENES_DIR=scenes/yemian/${tank_name}/rotategravity/$name
OUTPUT_DATA_DIR=scenes/zst/yemian/${tank_name}/rotategravity/$name
OUTPUT_NPZ_DIR=scenes/test/yemian/${tank_name}/rotategravity/${name}
OUTPUT_NPZ_FILE=scenes/test/yemian/${tank_name}/rotategravity/${name}/fluid_${frame}.npz
OUTPUT_BGEO_FILE=fluid_${frame}.bgeo
OUTPUT_OBJ_FILE=surface_e${frame}.obj

mkdir $OUTPUT_SCENES_DIR
mkdir $OUTPUT_DATA_DIR
mkdir $OUTPUT_NPZ_DIR
for seed in `seq 0 0`; do
    SIM_DIR="$OUTPUT_SCENES_DIR/sim_$(printf "%04d" $seed)"
    PARTIO_DIR="$OUTPUT_SCENES_DIR/sim_$(printf "%04d" $seed)/partio"
    count=0
    while [ ! -d "$PARTIO_DIR" ]; do
     # 如果 count 大于 0，则删除sim文件夹
      if [ $count -gt 0 ]; then
          rm -rf "$SIM_DIR"
          echo "Deleted directory $SIM_DIR"
      fi
      # generate raw data
      python datasets/create_physics_scenes_fuel_yemian_rotategravity.py --output $OUTPUT_SCENES_DIR \
                                              --seed $seed \
                                              --rotated_obj_path=/home/cyu/code/dual_small/scenes/fuel_triangletank/rotated_obj/yemian/${tank_name}/rotategravity \
                                              --henggun $henggun \
                                              --fuyang $fuyang
      count=$((count + 1))
      # Sleep for a short duration to avoid potential infinite fast looping
      sleep 1
    done
    echo "Directory $OUTPUT_DIR exists, moving to the next seed..."
done

# raw to zst
python datasets/create_physics_records.py --input $OUTPUT_SCENES_DIR \
                                 --output $OUTPUT_DATA_DIR

# Transforms and compresses the data such that it can be used for training.
# This will also create the OUTPUT_DATA_DIR.
for seed in `seq 0 0`; do
    OUTPUT_NPZ_DIR_SEED=${OUTPUT_NPZ_DIR}_${seed}
    # zst to npz
    python scripts/run_network_fueltank_zst_rotategravity.py --weights 52001_model_weights_best.pt \
                                            --scene scripts/example_scene.json \
                                            --zst_dir $OUTPUT_DATA_DIR \
                                            --output ${OUTPUT_NPZ_DIR_SEED} \
                                            --num_steps 400 \
                                            --sim_seed ${seed}
    # npz to bgeo
#    python npz2bgeo.py --path $OUTPUT_NPZ_FILE
    python npz2bgeo.py --path ${OUTPUT_NPZ_DIR_SEED}/fluid_${frame}.npz

    # bgeo to obj
#    cd $OUTPUT_NPZ_DIR
    splashsurf reconstruct ${OUTPUT_NPZ_DIR_SEED}/${OUTPUT_BGEO_FILE} -r=0.023 -l=1.8 -c=0.5 -t=0.6 --subdomain-grid=on --mesh-cleanup=on \
                --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10 -o ${OUTPUT_NPZ_DIR_SEED}/${OUTPUT_OBJ_FILE}
done

# Split data in train and validation set
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
