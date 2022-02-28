

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=64 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True --batch_size=64

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=256 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True --batch_size=64

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=512 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True --batch_size=128

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=128 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True --batch_size=128

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae  --epochs=100 --pretrain_epoch=0 --latent_dim=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True --batch_size=128
