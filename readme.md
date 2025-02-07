# README

## Installation Guide

### Step 1: Install Miniconda on macOS

To install Miniconda on macOS, open a terminal and run the following commands:

```sh
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### Step 2: Initialize Conda

After installation, close and reopen your terminal application or refresh it by running the following command:

```sh
source ~/miniconda3/bin/activate
```

To initialize Conda on all available shells, run the following command:

```sh
conda init --all
```

### Step 3: Create and Activate a New Conda Environment

To create a new Conda environment, replace `test` with your desired environment name:

```sh
conda create -n test python=3.9
```

Activate the environment:

```sh
conda activate test
```

### Step 4: Clone the repository
```sh
git clone https://github.com/zhuchen96/cell_data_extractor.git
cd cell_data_extractor
```

### Step 5: Install Required Packages

Install necessary dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Step 1: Activate the Conda Environment
```sh
conda activate test
```

### Step 2: Go to the curent folder
```sh
cd cell_data_extractor
```

### Step 3: Execute preparation software (only needed for the first time)
```sh
./preprocessing_mem.sh
```

### Step 4: Run the user interface
```sh
streamlit run main_page_script.py 
```
