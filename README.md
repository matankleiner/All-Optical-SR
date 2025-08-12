# All Optical Super Resolution

### Official pytorch implementation of the paper: "Can the Success of Digital Super-Resolution Networks Be Transferred to Passive All-Optical Systems?" 

## System Requirements 

### Hardware Requirements 

This code has been tested on a Linux machine (Ubuntu 22.04.4 LTS) with NVIDIA GeForce GTX 2080 Ti GPU. 

### Software Requirements 

This code has the following dependencies: 
```
  python >= 3.8.12 
  torch >= 1.12.1
  torchvision >= 0.13.1
  numpy >= 1.23.4
  tqdm >= 4.64.0
  torchmetrics >= 1.4.0
```

#### Setting an environment 

Create a python virtual environment, install all dependecies using the `requirements.txt` file and then run the code on your computer. 

```
cd DIR_NAME
python3 -m venv VENV_NAME
source VENV_NAME/bin/activate
pip install -r requirements.txt 
```
Installation time should take around 10 minutes. 

## Usage Instructions  

After installation one can run our code. 

### Data

The data used in our work is the MNIST, FashionMNIST, Quick, Drae!, KMNIST and EMNIST datasets. 

The MNIST, FashionMNIST and KMNIST datasets are available via `torchvision`. Quick, Draw! is avilable via their official [github](https://github.com/googlecreativelab/quickdraw-dataset). 
EMNIST dataset is available via their [official website](https://www.nist.gov/itl/products-and-services/emnist-dataset). 

For both the KMNIST and EMNIST datasets we used the drop-in replacemnt for the MNIST dataset. 

### Hyperparameters

`config.py` include all the hyperparameters used for each trial. The different hyperparameters used for running different experiemnts are detailed in the paper. 

### Usage

```
python3 main_trials.py --epochs 1000 --lr 1e-1 --trial_name unique_trial_name

```

## Licence 

Our code is under the MIT License. 

### Citation 

If you use this code for your research, please cite our paper:

```
```
