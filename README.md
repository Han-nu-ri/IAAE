# IAAE

dataset별 실험 : FFHQ


 - main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=8 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
 - main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=mask_aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=8 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
 - main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aaae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=8 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
 - main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=learning-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=8 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
 - main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=8 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
