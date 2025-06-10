&nbsp;

<div align="center">

<h2> Real-Time Infrared Denoising Mamba with Progressive Self Distillation  </h2> 

Yuchen Bai, Mingxin Yu, Weiqiang Chen, Lidan Lu, Xiaoping Lou, Yanlin He, Mingli Dong,
Lianqing Zhu 



</div>

## 1. Create Environment:

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to set up an environment.

``` sh
# Create environment
conda create -n RISE_Mamba python=3.11
conda activate RISE_Mamba

# Install pytorch 
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install open-mmlab packages
pip install -U openmim
mim install mmcv==2.1.0
mim install mmengine==0.10.4
mim install mmagic==1.2.0 

NOTE：If there are any issues with installing the open-mmlab packages, please refer to
 · https://github.com/open-mmlab/mmcv
 · https://github.com/open-mmlab/mmagic

# Install other packages
pip install -r requirements.txt
```



&nbsp;

## 2. Prepare Dataset:

Download our processed datasets from [Google drive](https://drive.google.com/file/d/1ytcmaj_Niv_EVMH10EKLTiUhODIlb2r9/view?usp=sharing),  [Baidu disk](https://pan.baidu.com/s/13rxgKvVXvZo3L2O6KvOmQg?pwd=13zp). Then put the downloaded datasets into the folder `data/` as

```sh
  |--data
      |--test_MP4
      |--test_PNG
      |--train_MP4
      |--train_PNG
```

&nbsp;

## 3. Testing:

```sh
python test.py 
```

&nbsp;

## 4. Training:


```sh
python train.py 
```


&nbsp;





