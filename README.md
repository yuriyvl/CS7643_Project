# MRI Puns Project

## Outline

### Dependencies and Installation

We have tested this code using:

* Ubuntu 18.04 / Windows 10 / Mac OS X (Catalina)
* Python 3.8
* CUDA 10.1
* CUDNN 7.6.5
* 1 / 4 CUDA-enabled GPUs

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

Navigate inside the `experimental/<model_name>/` folder and run the demo file inside with arguments. For example:

```bash
cd experimental
cd unet
python train_uret_demo.py --data_path ..\..\data --mode test --recon True
```
`--data_path` specify the location of the file to test with.
`--mode` specify the type which is test in our example.
`--recon` specify the reconstruction parameter.

The python script will automatically pick the files to test under the specified directory from singlecoil_test directory.

If the reconstruction is successful, the images will be placed under <file_name> directory under `experimental/<model_name>/`

In our example, the file name is file1000000.h5 which has 36 slices. We pick slice 22 for reconstruction because it resembles a complete knee.

The images stored are the input image to the model(file1000000.h5_22_image), the output image from the model(file1000000.h5_22_output) and the target image(file1000000.h5_22_target).

### Code References

The bulk of this repository is from https://github.com/facebookresearch/fastMRI. We made alterations/augmentations and added new models to the codebase.
