# 10-split CIFAR-100
GPUID=3
OUTDIR=outputs/CIFAR100_10tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py --agent_type dfd --agent_name DFD --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-5   --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 30 60 80 --batch_size 32 --model_name resnet18 --model_type resnet --model_lr 1e-3  --head_lr 5e-3 --reg_coef 20  | tee ${OUTDIR}/reg20_lr1e-3head_lr5e-3_wd5e-5_bs32_epoch306080.log


# 20-split CIFAR-100
GPUID=3
OUTDIR=outputs/CIFAR100_20tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py --agent_type dfd --agent_name DFD --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-4   --force_out_dim 0  --first_split_size 5 --other_split_size 5 --schedule 30 60 80 --batch_size 32 --model_name resnet18 --model_type resnet --model_lr 1e-4  --head_lr 1e-2 --reg_coef 10  | tee ${OUTDIR}/reg10_lr1e-4head_lr1e-2_wd5e-4_bs32_epoch306080.log


# 25-split TinyImageNet
GPUID=3
OUTDIR=outputs/TinyImageNet_25tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py --agent_type dfd --agent_name DFD --dataset TinyImageNet --dataroot ../../data/tiny-imagenet-200/ --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 1e-5   --force_out_dim 0  --first_split_size 8 --other_split_size 8 --schedule 30 60 80 --batch_size 16 --model_name resnet18 --model_type resnet --model_lr 5e-5  --head_lr 5e-3 --reg_coef 50  | tee ${OUTDIR}/reg50_lr5e-5head_lr5e-3_wd1e-5_bs16_epoch306080.log


# 20-split miniImageNet
GPUID=3
OUTDIR=outputs/mini_20tasks
mkdir -p $OUTDIR
REPEAT=1

python -u main.py --agent_type dfd --agent_name DFD --dataset miniImageNet --dataroot ../data/miniImageNet --gpuid $GPUID --repeat $REPEAT --model_optimizer Adam --model_weight_decay 5e-5 --force_out_dim 0  --first_split_size 5 --other_split_size 5 --schedule 30 60 80 --model_lr 5e-5 --head_lr 5e-3 --batch_size 32 --model_name reducedR18 --model_type resnet --reg_coef 20 | tee ${OUTDIR}/reg20_lr5e-5_headlr5e-3_wd5e-5_bs32_epoch306080.log

# 10-split SubImageNet
GPUID=3
OUTDIR=outputs/Sub_10tasks
REPEAT=1
mkdir -p $OUTDIR
python -u main.py --agent_type dfd --agent_name DFD --dataset SubImageNet --dataroot ../../../data/SubImageNet --gpuid $GPUID --repeat $REPEAT  --model_optimizer Adam --model_weight_decay 5e-5   --force_out_dim 0  --first_split_size 10 --other_split_size 10 --schedule 30 60 80 --batch_size 16 --model_name resnet50 --model_type resnet --model_lr 1e-4  --head_lr 1e-2 --reg_coef 200  | tee ${OUTDIR}/reg200_lr1e-4head_lr1e-2_wd5e-5_bs16_epoch306080.log

