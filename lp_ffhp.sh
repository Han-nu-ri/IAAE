

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=mask_aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aaae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=512 --mapper_inter_nz=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=learning-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=512 --mapper_inter_nz=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=512 --mapper_inter_nz=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True


python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=mask_aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aaae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=learning-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True



python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=mask_aae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --mapper_inter_nz=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aaae --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --mapper_inter_nz=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128  --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=learning-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --mapper_inter_nz=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --batch_size=64 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --mapper_inter_nz=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True