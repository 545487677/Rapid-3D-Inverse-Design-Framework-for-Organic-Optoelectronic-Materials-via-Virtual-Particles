## Environment with Docker Image

We provide a Docker image, which you can pull with the following command:
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

To use GPUs within Docker, you need to first install [`nvidia-docker-2`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Also, please install other environments

pip install rdkit

pip install lmdb

pip install py3Dmol

conda install -y -c conda-forge openbabel




## Usage Guide

### Baseline_Generated

We provide the OPV molecules generated based on the baseline method (stored in zip form at Baseline_Generated)

### Molecule Generation

To generate molecules, use the following command:

```bash
cd scripts
bash infer_opv_with_pretrain.sh 1.0 10000 2 0.1 1
```

### Molecule Evaluation

After generating molecules, you can evaluate them using the `O2_GEN_Metric.py` script located in the `OPV_INFER_EVAL` folder:

```bash
cd OPV_INFER_EVAL
python O2_GEN_Metric.py
```

We also provide several files for replicating baseline results for performance comparison.

### Molecule Screening

We offer various screening methods to suit different application needs:

#### General Screening

Perform general screening with the following command:

```bash
cd scripts
bash infer_opv_screening.sh 1.0 20000 2 0.006 todft tolmdb 0 0 520 520
```

#### Screening by Gap Type

Specify minimum (min) or maximum (max) gap for targeted screening:

```bash
cd scripts
bash infer_opv_screening.sh 1.0 20000 2 0.006 todft tolmdb min 0 520 520
# Or
cd scripts
bash infer_opv_screening.sh 1.0 20000 2 0.006 todft tolmdb max 0 520 520
```

#### Specific Screening (Strict Filter)

Conduct specific screening with strict filtering:

```bash
cd scripts
bash infer_opv_screening.sh 1.0 20000 2 0.006 todft tolmdb 0 strict_filter 520 520
```


#### Specific Screening DB
We also provide the database after specific screening at the Specific_General_DB.