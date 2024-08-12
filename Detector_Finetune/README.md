## Environment with Docker Image

We provide a Docker image, which you can pull with the following command:
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

To use GPUs within Docker, you need to first install [`nvidia-docker-2`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Using the Scripts

Navigate to the path of the scripts folder:

cd path/to/scripts

train detector: bash train.sh 1e-4 32 1000 0 0.06 9 0

valid_finetune_dft: bash valid_finetune.sh

valid_generel_screen: bash valid_general.sh

valid_optimization_screen: bash valid_opt.sh
