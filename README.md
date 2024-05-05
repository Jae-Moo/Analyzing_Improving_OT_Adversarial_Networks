# Analyzing and Improving Optimal-Transport-based Adversarial Networks (ICLR, 2024) #
Link to paper: https://arxiv.org/abs/2310.02611

## Analyzing OT-based GANs
#### Toy ####
Commands for toy experiments are as follows.
```
python train.py --dataset {spiral,8gaussian,25gaussian,checkerboard} --image_size 2 --nz 2 --model_name toy --tau {0,1} --lmbda 0 --num_iterations 50000 --beta1 {0,0.5} --save_content_every 5000 --save_ckpt_every 5000 --save_image_every 2000
```
Example 1
```
python train.py --exp UOTM_lmbda0.0beta0.5tau1 --dataset 8gaussian --image_size 2 --nz 2 --model_name toy --tau 1 --num_iterations 50000 --save_content_every 5000 --save_ckpt_every 5000 --save_image_every 2000 --phi1 kl --phi2 kl --phi3 linear --reg_name none --lmbda 0.0 --beta1 0.5
```

#### CIFAR-10 ####
We train UOTM on CIFAR-10 using 4 32-GB V100 GPU. 

WGAN-GP
```
python train.py --exp WGAN-GP --phi1 linear --phi2 linear --phi3 linear --model_name {ncsnpp,otm} --batch_size 256 --beta1 0 --tau 0 --reg_name gp --lmbda 10 --use_ema --lr_scheduler
```

WGAN-R1
```
python train.py --exp WGAN-R1 --phi1 linear --phi2 linear --phi3 linear --model_name {ncsnpp,otm} --batch_size 256 --beta1 0 --tau 0 --reg_name r1 --lmbda 0.2 --use_ema --lr_scheduler
```

OTM
```
python train.py --exp OTM --phi1 linear --phi2 linear --phi3 linear --model_name {ncsnpp,otm} --batch_size 256 --beta1 0 --tau 0.001 --reg_name r1 --lmbda 0.2 --use_ema --lr_scheduler
```

UOTM
```
python train.py --exp UOTM --phi1 {kl,chi,softplus} --phi2 {kl,chi,softplus} --phi3 linear --model_name {ncsnpp,otm} --batch_size 256 --beta1 0.5 --tau 0.001 --reg_name r1 --lmbda 0.2 --use_ema --lr_scheduler
```


## UOTM with Scheduled Divergence (UOTM-SD) ##
We train UOTM-SD on CIFAR-10 using 4 32-GB V100 GPU. 
```
python train.py --exp {exp_name} --loss_scheduler linear --alpha_min 0.2 --alpha_max 5 --schedule_until 150000 --cost w2 --phi1 softplus --phi2 softplus --phi3 linear --model_name ncsnpp --ngf 64 --batch_size 256 --beta1 0.5 --tau 0.001 --lmbda 0.2 --use_ema --num_iterations 200000 --lr_g 2e-4
```

## Bibtex ##
Cite our paper using the following BibTeX item:
```
@inproceedings{choi2023analyzing,
  title={Analyzing and Improving Optimal-Transport-based Adversarial Networks},
  author={Choi, Jaemoo and Choi, Jaewoong and Kang, Myungjoo},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```