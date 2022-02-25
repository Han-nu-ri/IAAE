

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=16 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --epochs=100 --pretrain_epoch=0 --latent_dim=16 --mapper_inter_nz=16 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True


python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --epochs=100 --pretrain_epoch=0 --latent_dim=64 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior  --epochs=100 --pretrain_epoch=0 --latent_dim=64 --mapper_inter_nz=64 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --epochs=100 --pretrain_epoch=0 --latent_dim=256 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior  --epochs=100 --pretrain_epoch=0 --latent_dim=256 --mapper_inter_nz=16 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=256 --time_check=True