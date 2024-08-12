## Environment with Docker Image

We provide a Docker image, which you can pull with the following command:
docker pull dptechnology/unicore:latest-pytorch2.0.1-cuda11.7-rdma

To use GPUs within Docker, you need to first install [`nvidia-docker-2`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Using the Scripts

Navigate to the path of the scripts folder:

cd path/to/scripts

change the data path in the shell

bash pretrain_electro.sh 16 1

Although the training step is 2,000,000 steps, we halted the training at 990,542 steps, which corresponded to 48 epochs. We used 12 V100 32G to traing the model.
