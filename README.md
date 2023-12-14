### Setup
- Get Code
```shell
 git clone https://github.com/Garsonjw/FaceCat.git
```
- Build Environment
```shell
cd FaceCat
conda create -n fas python=3.9
pip install -r requirements.txt
conda activate fas
```

### Dataset
SiW-Mv2: https://github.com/CHELSEA234/Multi-domain-learning-FAS?tab=readme-ov-file.

Data preprocessing: run [video_to_crop.py](https://github.com/Garsonjw/FaceCat/blob/main/video_to_crop.py) and change the address of SiW-Mv2.

### Training and Inference
Before training, please make sure to add the ffhq.pt to the corresponding address
