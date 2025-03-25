# DualFluidNet: An attention-based dual-pipeline network for fluid simulation

This repository contains code for our *Neural Networks* 2024 paper. 
We show how to train particle-based fluid simulation networks as CNNs using 
continuous convolutions. The code allows you to generate data, train your own 
model or just run a pretrained model.

Please cite our paper [(pdf)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003253) if you find this code useful:
```
@article{CHEN2024106401,
title = {DualFluidNet: An attention-based dual-pipeline network for fluid simulation},
journal = {Neural Networks},
volume = {177},
pages = {106401},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106401},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024003253},
author = {Yu Chen and Shuai Zheng and Menglong Jin and Yan Chang and Nianyi Wang},
keywords = {Fluid simulation, Learning physics, Neural network, Deep learning},
abstract = {Fluid motion can be considered as a point cloud transformation when using the SPH method. Compared to traditional numerical analysis methods, using machine learning techniques to learn physics simulations can achieve near-accurate results, while significantly increasing efficiency. In this paper, we propose an innovative approach for 3D fluid simulations utilizing an Attention-based Dual-pipeline Network, which employs a dual-pipeline architecture, seamlessly integrated with an Attention-based Feature Fusion Module. Unlike previous methods, which often make difficult trade-offs between global fluid control and physical law constraints, we find a way to achieve a better balance between these two crucial aspects with a well-designed dual-pipeline approach. Additionally, we design a Type-aware Input Module to adaptively recognize particles of different types and perform feature fusion afterward, such that fluid-solid coupling issues can be better dealt with. Furthermore, we propose a new dataset, Tank3D, to further explore the network’s ability to handle more complicated scenes. The experiments demonstrate that our approach not only attains a quantitative enhancement in various metrics, surpassing the state-of-the-art methods, but also signifies a qualitative leap in neural network-based simulation by faithfully adhering to the physical laws. Code and video demonstrations are available at https://github.com/chenyu-xjtu/DualFluidNet.}
}
```

## Dependencies

- PyTorch 1.13.0+cu116 (```pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116``` )
- Open3D 0.17.0 (```pip install open3d==0.17.0``` )
- SPlisHSPlasH 2.4.0 (for generating training data and fluid particle sampling, https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- Tensorpack DataFlow (for reading data, ```pip install --upgrade git+https://github.com/tensorpack/dataflow.git```)
- python-prctl (needed by Tensorpack DataFlow; depends on libcap-dev, install with ```apt install libcap-dev``` )
- msgpack (```pip install msgpack``` )
- msgpack-numpy (```pip install msgpack-numpy```)
- python-zstandard (```pip install zstandard``` https://github.com/indygreg/python-zstandard)
- partio (https://github.com/wdas/partio)
- SciPy
- OpenVDB with python binding (optional for creating surface meshes, https://github.com/AcademySoftwareFoundation/openvdb)
- plyfile (optional for creating surface meshes, ```pip install plyfile```)
- others (```pip install -r requirements.txt```)
The versions match the configuration that we have tested on a system with Ubuntu 18.04.
SPlisHSPlasH 2.4.0 is required for generating training data (ensure that it is compiled in *Release* mode).
We recommend to use the latest versions for all other packages.

## Datasets

### 1. Water block dataset download
If you want to skip the data generation step you can download training and validation data from the links below.

| water default data  | 34GB | [link](https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07) |

For the default data the training set has been generated with the scripts in this
repository and the validation data corresponds to the data used in the paper.

### 2. Fuel surface dataset generation
The data generation scripts are in the ```datasets``` subfolder.
To generate the training and validation data 
 1. Set the path to the ```DynamicBoundarySimulator``` of SPlisHSPlasH in the ```datasets/splishsplash_config.py``` script.
 2. Run the scripts from within the datasets folder [raw directories -> .zst -> .npz (-> .obj)]

If you want to rotate the direction of the box, and gravity stays unchanged:
    ```
    sh datasets/create_fuel_yemian_rotatebox.sh
    ```

If you want to rotate the direction of gravity, and the box stays unmoved:
    ```
    sh datasets/create_fuel_yemian_rotategravity.sh
    ```

### 3. (Optional) If you want to run the generated zst files directly
```bash 
# run the exist zst files
scripts/run_network_fueltank.py --weights=ckpts/52001_model_weights_best.pt \
                --scene=scripts/example_scene.json \
                --zst_dir=[zst directory] \
                --output=[output directory] \
                --num_steps=600
```
Here, ```--weights``` and ```--scene``` make no difference (i.e. 在这里这两个参数不用管，但代码中没有删掉这两个参数所以运行时需要带上，可以自行修改代码). This script just run the existed zst files which is set by ```--zst_dir```. The output path is set by ```--output```.

## Training the network

### Training scripts
To train the model with the generated data simply run one of the ```scripts/train_network_torch.py``` scripts from within the ```scripts``` folder. 
```bash
# PyTorch version
scripts/train_network_torch.py default.yaml
```

```default.yaml``` is the training configuration, including the dataset path and whether rotation should be applied during training.

The scripts will create a folder ```train_network_torch_default``` respectively with snapshots and log files.
The log files can be viewed with Tensorboard.

If you want to train on other dataset, change the ```dataset_dir``` in ```default.yaml```

```ckpts/ckpt-52000.pt``` is the best checkpoint pretrained on the default water block dataset. If you want to train on fuel datasets, it is recommended to train on the checkpoint pretrained on water datasets: 
Copy the ckpts/ckpt-52000.pt into the folder ```train_network_torch_default/checkpoints```, then the training will start by the step 52000.

### Evaluating the network
To evaluate the network run the ```scripts/evaluate_network.py``` script like this
```bash
scripts/evaluate_network.py --trainscript=scripts/train_network_torch.py \
        --cfg=scripts/default.yaml \
        --weights=/home/cyu/code/dual_small/ckpt/69R_tank_yemian/ckpt-59000.pt
```
```--trainscript``` is the corresponding training script,  ```--cfg``` is the evaluating configuration, and ```--weights``` is the path to the weight (checkpoint) that is to be evaluated.

This will create the file ```train_network_torch_default_eval_50000.json```, which contains the 
individual errors between frame pairs.

The script will also print the overall errors. The output should look like 
this if you use the generated the data:
```{'err_n1': 0.0005267939744493333, 'err_n2': 0.0014583940438080846, 'emd_n1': 9.877338082988465e-05, 'emd_n2': 0.00014583940438080846,'whole_seq_err': 0.029428330705331872}```

## Running the pretrained model

The pretrained network weights are in ```ckpts/52001_model_weights_best.pt``` for PyTorch.
The following code runs the network on the example scene
```bash 
# run the pretrained model for single fluid
scripts/run_network.py ---weights=ckpts/52001_model_weights_best.pt \
                --scene=scenes/example_scene.json \
                --output=scenes/output/test \
                --num_steps=250 \
                train_network_torch.py
# run the pretrained model for multi fluids    
scripts/run_network_multiflulid.py --weights=ckpts/52001_model_weights_best.pt \
                --scene=scenes/example_scene.json \
                --output=scenes/output.test \
                --num_steps=300 \
                train_network_torch.py 
```

The script writes point clouds with the particle positions as .ply files, which can be visualized with Open3D.
Note that SPlisHSPlasH is required for sampling the initial fluid volumes from ```.obj``` files.

## Rendering

See the [scenes](scenes/README.md) directory for instructions on how to create and render the example scenes like the canyon.

## Licenses

Code and scripts are under the MIT license.