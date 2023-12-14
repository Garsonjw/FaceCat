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

### Training
Before training, please make sure to add the ffhq.pt to [/checkpoints/ddpm/](https://github.com/Garsonjw/FaceCat/tree/12306a2988432038f82d2e24599c59bf81ece967/checkpoints/ddpm). Then, you can run the following script for training：
```shell
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port=12349 train_fas.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --use_fp16 True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --exp experiments/ffhq_34/ddpm.json --clip
```

### Inference
If you already have a checkpoint(fas_model_p1_best.pth), please place it at the [/checkpoints/](https://github.com/Garsonjw/FaceCat/tree/12306a2988432038f82d2e24599c59bf81ece967/checkpoints/) in the root directory. Then, you can run the following script for testing：
```shell
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port=12349 test_fas.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --use_fp16 True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --exp experiments/ffhq_34/ddpm.json --clip
```
If you want to test the effect of baseline, please execute:
```shell
cd face_anti_spoofing
```
Then, add the CDCN_test_Best(epoch=20).pt, CMFL_test_Best(epoch=11).pt and depthnet_test_Best(epoch=16).pt to [/checkpoints/](https://github.com/Garsonjw/FaceCat/tree/12306a2988432038f82d2e24599c59bf81ece967/checkpoints/ddpm)
