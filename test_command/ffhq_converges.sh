

python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --epochs=500 --pretrain_epoch=0 --latent_dim=128 --log_interval=10 --logging_start=100 --stop_by_fid=True --stop_patient=3 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128

python3 main.py --device=cuda:1 --dataset=ffhq --image_size=32 --model_name=mask_aae --epochs=500 --pretrain_epoch=0 --latent_dim=128 --log_interval=10 --logging_start=100 --stop_by_fid=True --stop_patient=3 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128

python3 main.py --device=cuda:1 --dataset=ffhq --image_size=32 --model_name=aaae --epochs=500 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=10 --logging_start=100 --stop_by_fid=True --stop_patient=3 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128

python3 main.py --device=cuda:1 --dataset=ffhq --image_size=32 --model_name=learning-prior --epochs=500 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=10 --logging_start=100 --stop_by_fid=True --stop_patient=3 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128

python3 main.py --device=cuda:1 --dataset=ffhq --image_size=32 --model_name=non-prior --epochs=500 --pretrain_epoch=0 --latent_dim=128 --mapper_inter_nz=128 --log_interval=10 --logging_start=100 --stop_by_fid=True --stop_patient=3 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128
