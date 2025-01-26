---
title: TraceFL Interpretability-Driven Debugging in Federated Learning via Neuron Provenance # TODO
url: https://arxiv.org/pdf/2312.13632 # TODO: update with the link to your paper
labels: [interpretability, Explainability, Transparency, Trustworthy, Healthcare, Clients Importance, Text Classification, Image Classificatoin] # TODO: please add between 4 and 10 single-word (maybe two-words) labels (e.g. system heterogeneity, image classification, asynchronous, weight sharing, cross-silo). Do not use "". Remove this comment once you are done.
dataset: [cifar10, mnist, pathmnist, organamnist, dbpedia_14, yahoo_answers_topics] # TODO: list of datasets you include in your baseline. Do not use "". Remove this comment once you are done.
---

> [!IMPORTANT]
> This is the template for your `README.md`. Please fill-in the information in all areas with a :warning: symbol.
> Please refer to the [Flower Baselines contribution](https://flower.ai/docs/baselines/how-to-contribute-baselines.html) and [Flower Baselines usage](https://flower.ai/docs/baselines/how-to-use-baselines.html) guides for more details.
> Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.
> Please remove this [!IMPORTANT] block once you are done with your `README.md` as well as all the `:warning:` symbols and the comments next to them.

> [!IMPORTANT]
> To help having all baselines similarly formatted and structured, we have included two scripts in `baselines/dev` that when run will format your code and run some tests checking if it's formatted.
> These checks use standard packages such as `isort`, `black`, `pylint` and others. You as a baseline creator will need to install additional pacakges. These are already specified in the `pyproject.toml` of 
> your baseline. Follow these steps:

```bash
# Create a python env
pyenv virtualenv 3.10.14 tracefl

# Activate it
pyenv activate tracefl

# Install project including developer packages
# Note the `-e` this means you install it in editable mode 
# so even if you change the code you don't need to do `pip install`
# again. However, if you add a new dependency to `pyproject.toml` you
# will need to re-run the command below
pip install -e ".[dev]"

# Even without modifying or adding new code, you can run your baseline
# with the placeholder code generated when you did `flwr new`. If you
# want to test this to familiarise yourself with how flower apps are
# executed, execute this from the directory where you `pyproject.toml` is:
flwr run .

# At anypoint during the process of creating your baseline you can 
# run the formatting script. For this do:
cd .. # so you are in the `flower/baselines` directory

# Run the formatting script (it will auto-correct issues if possible)
./dev/format-baseline.sh tracefl

# Then, if the above is all good, run the tests.
./dev/test-baseline.sh tracefl
```

> [!IMPORTANT]
> When you open a PR to get the baseline merged into the main Flower repository, the `./dev/test-baseline.sh` script will run. Only if test pass, the baseline can be merged. 
> Some issues highlighted by the tests script are easier than others to fix. Do not hesitate in reaching out for help to us (e.g. as a comment in your PR) if you are stuck with these.
> Before opening your PR, please remove the code snippet above as well all the [!IMPORTANT] message blocks. Yes, including this one.

# TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance

> [!NOTE] 
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://arxiv.org/abs/2312.13632

**Authors:** : [Waris gill](https://people.cs.vt.edu/waris/), [Ali Anwar](https://chalianwar.github.io/), [Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/)

**Abstract:** In Federated Learning, clients train models on local data and send updates to a central server, which aggregates them into a global model using a fusion algorithm. This collaborative yet privacy-preserving training comes at a cost. FL developers face significant challenges in attributing global model predictions to specific clients. Localizing responsible clients is a crucial step towards (a) excluding clients primarily responsible for incorrect predictions and (b) encouraging clients who contributed high-quality models to continue participating in the future. Existing ML debugging approaches are inherently inapplicable as they are designed for single-model, centralized training.

We introduce TraceFL, a fine-grained neuron provenance capturing mechanism that identifies clients responsible for a global model's prediction by tracking the flow of information from individual clients to the global model. Since inference on different inputs activates a different set of neurons of the global model, TraceFL dynamically quantifies the significance of the global model's neurons in a given prediction, identifying the most crucial neurons in the global model. It then maps them to the corresponding neurons in every participating client to determine each client's contribution, ultimately localizing the responsible client. We evaluate TraceFL on six datasets, including two real-world medical imaging datasets and four neural networks, including advanced models such as GPT. TraceFL achieves 99% accuracy in localizing the responsible client in FL tasks spanning both image and text classification tasks. At a time when state-of-the-artML debugging approaches are mostly domain-specific (e.g., image classification only), TraceFL is the first technique to enable highly accurate automated reasoning across a wide range of FL applications.


## About this artifact

**What’s implemented:**  We implmented the complete TraceFL framework. The code is available in the `tracefl` directory. We also provide a jupyter notebook that can easily run with a signle click on google colab that can be used to run the code. The notebook is available in the `artifact.ipynb` file. In orignal paper, TraceFL is evaluted on on n 20,600 cleintss models in total accross all combinations of datasets and models and reproducing these resutls took days even on high end cluster with A100 GPUs. 

First we provide a single command that can be used to run any of the experiments presented in the paper. Additiionlly, we provide a representative scripts for each figure and table to reproduce the foundational results. You can modify these scripts to run with different hyperparameters or datasets to produce remaiing results or same results in the figure.    



<!-- :warning: *_Concisely describe what experiment(s) (e.g. Figure 1, Table 2, etc) in the publication can be replicated by running the code. Please only use a few sentences. ”_* -->



<!-- :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset). We highly recommend using [FlowerDatasets](https://flower.ai/docs/datasets/index.html) to download and partition your dataset. If you have other ways to download the data, you can also use `FlowerDatasets` to partiion it._* -->

**Hardware Setup:**  
- **Orginal Paper Hardware Setup**: To resemble real-world FL and do large scale simulations, we deploy our experiments in [Flower FL framework](https://flower.ai/), running on an enterprise-level cluster of six NVIDIA DGX A100 nodes. Each node is equipped with 2048 GB of memory, at least 128 cores, and an A100 GPU with 80 GB of memory.

- **Artifact Hardware Setup**: We change the default configuration in `tracefl/conf/base.yaml` to run representative experiments on Google Colab even with only 2 cpu cores, 12 GB of System RAM and 15 GB of GPU RAM. 


<!-- **Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_* -->


## Experimental Setup

**Task:** The goal is to achieve interpretability in federated learning. TraceFL addresses the interpretability problem in FL: **Given the global model inference on an input in FL, how can we identify the client(s) most responsible for the inference?**

**Model:** 
- **Text Classification Models**: `GPT`, `BERT` 
- **Image Classificaiton Modesl**: `ResNet`, `DenseNet`

<!-- :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._* -->

**Datasets:** 
- **Image Classification**: `Cifar10`, `Mnist`  
- **Medical Image Classification**: `Colon-Pathology`, `Abdominal-CT` 
- **Text Classification**:  `DBpedia`, `Yahoo-Answers`

We use `FlowerDatasets` to configure the datasets and partitions. Our code is designed with any classfication datasets (image, text) available in `FlowerDatasets`  or configured with `FlowerDatasets` provided guidliens. 

<!-- **Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._* -->

**Training Hyperparameters:** Main hyperparameters are listed in the `tracefl/conf/base.yaml` file and can be adjusted to run different experiments.

<!-- :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._* -->
```yaml
num_clients: 10 
clients_per_round: 10
num_rounds: 10
dirichlet_alpha: None # non-iid
batch_size: 32 # client batch size
# for differential privacy
noise_multiplier: -1
clipping_norm: -1

total_gpus: 1
total_cpus: 2
client_cpus: 2
client_gpu: 1
device: "cpu" # "cpu" or "cuda"

client:
    epochs: 2
    lr: 0.001

model:
    name: densenet121 #google-bert/bert-base-cased  #microsoft/MiniLM-L12-H384-uncased #Intel/dynamic_tinybert #google-bert/bert-base-cased #resnet18
    arch: ${model_arch.${model.name}}

dataset:
    name: 'pathmnist' # organamnist
    num_classes: ${dataset_classes.${dataset.name}}
    channels: ${dataset_channels.${dataset.name}}

strategy:
    name: fedavg # points to your strategy (either custom or exiting in Flower)
    num_rounds: ${num_rounds}
    clients_per_round: ${clients_per_round}
    noise_multiplier: ${noise_multiplier}
    clipping_norm: ${clipping_norm}

data_dist:
    dist_type: non_iid_dirichlet #PathologicalPartitioner-3 # non_iid_dirichlet
    num_clients: ${num_clients}
    batch_size: ${batch_size}
    dirichlet_alpha: ${dirichlet_alpha}
    dname: ${dataset.name}
    mname: ${model.name}
    storage_dir: ${storage.dir}
    max_per_client_data_size: 2048
    max_server_data_size: 2048
```


## Environment Setup

:warning: _Specify the steps to create and activate your environment and install the baseline project. Most baselines are expected to require minimal steps as shown below. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._

:warning: _The dependencies for your baseline are listed in the `pyproject.toml`, extend it with additional packages needed for your baseline._

:warning: _Baselines should use Python 3.10, [pyenv](https://github.com/pyenv/pyenv), and the [virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugging. 

```bash
# create a new environment with Python 3.10 and install poetry. poetry is used to install the dependencies.
conda create --name tracefl python=3.10 -y
conda activate tracefl
pip install poetry

# cloning TraceFL repository and installing dependencies
git clone https://github.com/warisgill/TraceFL-Artifact.git
cd TraceFL-Artifact
mkdir logs

# install all the dependencies required for the artifact to run
poetry install
```

scuccessful installation will show the following message:

```output 
add mesage here
```



<!-- :warning: _If your baseline requires running some script before starting an experiment, please indicate so here_. -->

## Running the Experiments

Make sure you have adjusted the `client-resources` in  `tracefl/conf/base.yaml`  so your simulation makes the best use of the system resources available. Currently, the default configuration is set to run on a system with 2 cpu cores, 12 GB of System RAM and 15 GB of GPU RAM so you can run the experiments on Google Colab.




```bash

# this command will run the experiment with the default configuration present in the `tracefl/conf/base.yaml` file.
# you can modify the configuration file to run the experiment with different hyperparameters. or you pass the any of the hyperparameters as command line arguments

python -m tracefl.main 

```





