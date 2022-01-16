REM python aae_for_various_prior.py --dataset=ffhq32 --distribution=standard_normal
REM python aae_for_various_prior.py --dataset=ffhq32 --distribution=uniform
REM python aae_for_various_prior.py --dataset=ffhq32 --distribution=gamma
REM python aae_for_various_prior.py --dataset=ffhq32 --distribution=beta
python aae_likelihood.py --dataset=ffhq32 --distribution=chi
REM python aae_for_various_prior.py --dataset=ffhq32 --distribution=laplace

REM python aae.py --dataset=ffhq64
REM python aae.py --dataset=cifar
REM python aae.py --dataset=emnist
REM python aae.py --dataset=mnist
REM python aae.py --dataset=mnist_fashion
