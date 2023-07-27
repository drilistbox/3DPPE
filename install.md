# conda create -n 
```
#清华源
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.4.8
MMCV_WITH_OPS=1 pip install -e . -v
pip uninstall mmcv

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.0
pip install -v -e .

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.25.0
pip install -v -e .

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc3
pip install -v -e .

pip install einops

#注意
/root/miniconda3/lib/python3.8/site-packages/torch/optim/adamw.py
110行
F.adamw(params_with_grad,
要向后退一个tab
```



## Install MMCV
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
examples：
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
## Install MMDetection

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install MMSegmentation.

```bash
sudo pip install mmsegmentation==0.20.2
```

## Install MMDetection3D

```bash
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install PETR

```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts
mkdir data
ln -s {mmdetection3d_path} ./mmdetection3d
ln -s {nuscenes_path} ./data/nuscenes
```
examples
```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts ###pretrain weights
mkdir data ###dataset
ln -s ../mmdetection3d ./mmdetection3d
ln -s /data/Dataset/nuScenes ./data/nuscenes
```


# 数据准备
## 下载nus数据
```
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-mini
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
```

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

