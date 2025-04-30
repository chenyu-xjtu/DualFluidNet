# A Pioneering Neural Network Method for Efficient and Robust Fuel Sloshing Simulation in Aircraft
Our paper has been accepted by *AAAI* 2025 🔥🔥🔥
```
@inproceedings{chen2025pioneering,
  title={A Pioneering Neural Network Method for Efficient and Robust Fuel Sloshing Simulation in Aircraft},
  author={Chen, Yu and Zheng, Shuai and Wang, Nianyi and Jin, Menglong and Chang, Yan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={15},
  pages={15957--15965},
  year={2025}
}
```

Another related paper has been accepted by *Neural Networks* 🔥🔥🔥
```
@article{chen2024dualfluidnet,
  title={DualFluidNet: An attention-based dual-pipeline network for fluid simulation},
  author={Chen, Yu and Zheng, Shuai and Jin, Menglong and Chang, Yan and Wang, Nianyi},
  journal={Neural Networks},
  volume={177},
  pages={106401},
  year={2024},
  publisher={Elsevier}
}
```

![Fluid Simulation in Canyon](https://github.com/chenyu-xjtu/A-Pioneering-Neural-Network-Method-for-Efficient-and-Robust-Fuel-Sloshing-Simulation-in-Aircraft/blob/main/canyon.gif)

This repository contains code for our network for fluid simulation.
We show how to train particle-based fluid simulation networks as CNNs using 
continuous convolutions. The code allows you to generate data, train your own 
model or just run a pretrained model.

Please cite our paper [(pdf)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024003253) if you find this code useful:



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

| water default data  | 34GB | [link](https://ojs.aaai.org/index.php/AAAI/article/view/33752) |

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
