# MRI Puns Project

## Outline

### Dependencies and Installation

We have tested this code using:

* Ubuntu 18.04
* Python 3.8
* CUDA 10.1
* CUDNN 7.6.5
* Two CUDA-enabled GPUs

First install PyTorch according to the directions at the [PyTorch Website](https://pytorch.org/get-started/) for your operating system and CUDA setup.

Then, navigate to the root directory and run:

```bash
pip install -e .
```

`pip` will handle all package dependencies. After this you should be able to run most the code in the repository.

### Training the models

Navigate inside the `experimental/<model_name>/` folder and run the demo file inside with no arguments. For example:

```bash
cd experimental
cd nnret
python train_nnret_demo.py
```

### Graphing the training results

After training a model, take the console output, place it in a textfile and place this file inside a new folder. Then place the folder inside `results`. Alter the run() method in `results/grapher.py` to include the output and run:

```bash
python grapher.py
```

### Test using the models

Follow the same steps as training except call the demo file with `--mode test`. For example:

```bash
python train_nnret_demo.py --mode test
```

### Extracting reconstructed images from test output

TODO

### Code References

The bulk of this repository is from https://github.com/facebookresearch/fastMRI. We made alterations/augmentations and added new models to the codebase.
