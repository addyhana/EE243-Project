# EE243-Project

### To run experiments

#### 1. Clone this repository

```bash
git clone https://github.com/addyhana/EE243-Project.git
cd EE243-Project
```

#### 2. Clone the official SAM2 repository

From the base directory (`EE243-Project/`):

```bash
git clone https://github.com/facebookresearch/sam2.git
```

#### 3. Create and activate a Python environment

Using Conda:

```bash
conda create -n sam2 python=3.10 -y
conda activate sam2
```

#### 4. Install PyTorch

Install the version appropriate for your system and CUDA version. For example:

```bash
pip install torch torchvision
```

For CUDA-enabled systems, see the official PyTorch installation instructions:
https://pytorch.org/get-started/locally/

#### 5. Install SAM2

```bash
cd sam2
pip install -e .
```

If CUDA is available, install the optional CUDA extensions:

```bash
pip install -e ".[notebooks]"
```

#### 6. Install project dependencies

Return to the project root directory:

```bash
cd ..
```

Install the required packages:

```bash
pip install -r requirements.txt
```

#### 7. Run experiments

If running locally, example run: 

```bash
python run_sam2_baseline.py \
  --video_dir experiments/input_frames \
  --first_mask experiments/input_first_mask/first_mask.png \
  --output_dir results/dog_baseline
```

If running on UCR GPU cluster, run the appropriate .slurm file with appropriate destinations 

# ** TODO MAKE THIS EASIER 2 USE!! ** 

Generated masks and overlays will be saved to:

```text
results/[experiment#]/
```
