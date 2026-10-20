# Python

Python code to find a universal perturbation [[1]](http://arxiv.org/pdf/1610.08401), using the [PyTorch](https://pytorch.org/) library.

## Installation

```
pip install -r requirements.txt
```

The code runs on CUDA, Apple Silicon (MPS) or CPU, selected automatically.
`pyarrow` and `huggingface_hub` are also needed to download and convert ImageNet (`prepare_imagenet_val.py`).

## Files

| File | Role |
|---|---|
| `deepfool.py` | DeepFool [[2]](http://arxiv.org/pdf/1511.04599): minimal perturbation that changes the label of one image |
| `universal_pert.py` | `universal_perturbation`: the algorithm of [1], with optional checkpoint/resume |
| `prepare_imagenet_data.py` | Pre-processing (256 resize, 224 center crop, values in [0, 1]) and loading of N images per class |
| `demo.py` | Computes (or loads) the universal perturbation of a torchvision model, and shows it on a test image |
| `evaluate.py` | Evaluates a perturbation on held-out images: accuracy, fooling rate, random control, dominant labels, figure |
| `prepare_imagenet_val.py` | Converts the ImageNet validation set from Hugging Face into class folders, split into X (10/class) and eval (40/class) |
| `run_models.sh` | Computes and evaluates the perturbation of several models in a row (resumable) |
| `results/` | Results of the experiments ([README](results/README.md)) |
| `testlogistic/` | Tests of DeepFool ([README](testlogistic/README.md)) |

## Usage

### Get started

To get started, run the demo code with a pre-trained [torchvision](https://pytorch.org/vision/stable/models.html) model on the image of your choice
```
python demo.py -i data/test_img.png -t <training images path> -m resnet50
```
The training images path must contain one sub-folder per class (e.g. `n01440764/`, as in ImageNet or Imagenette).
If no universal perturbation has been computed for the model yet, it is computed on `--num_imgs_per_class` images per class
(the pre-processed images are cached in the `--cache` file) and saved in `data/universal_<model>.pt`.
The demo then shows the image without and with universal perturbation, with the estimated labels.
If `data/universal_<model>.pt` already exists, it is loaded instead: delete it to compute the perturbation again.

Options:
- `--xi`: l_inf magnitude of the perturbation on the [0, 255] scale (default 10)
- `--delta`: desired fooling rate = 1 - delta (default 0.2)
- `--num_imgs_per_class` (default 10)
- `--cache`: file where the pre-processed images are cached (default `data/imagenet_data.pt`). Use one file per dataset, otherwise the cache of another dataset is reused.
- `--max_iter_df`: maximum number of DeepFool iterations (default 10). The update is discarded when DeepFool does not converge; the number of discarded updates is printed after each pass.
- `-o`, `--perturbation`: file of the universal perturbation (default `data/universal_<model>.pt`). Use it to keep several perturbations of the same model (e.g. computed on different datasets).
- `--checkpoint`: file where the computation is saved every 200 images and after each pass (default `data/checkpoint_<model>.pt`, or `<perturbation>_checkpoint.pt` with `-o`). If the computation is interrupted, relaunching the same command resumes it. The file is deleted at the end.

### Evaluating a universal perturbation

```
python evaluate.py -m resnet50 -v <validation images path> [-p data/universal_resnet50.pt] [--out results/my_experiment]
```
It reports, on images that were not used to compute the perturbation: the accuracy without perturbation, with a control perturbation
(the pixels of v shuffled, same norms) and with the universal perturbation, the fooling rate, and the labels most predicted after perturbation.
With `--out`, the results and a figure of examples are saved in the given folder. `-p` makes it possible to evaluate the perturbation of one model
on another model (transferability). `--batch_size` (default 100) can be lowered on small GPUs, `--max_per_class` limits the number of images.

### Reproducing the protocol of the paper on ImageNet

```bash
# 1. Download the ImageNet validation set (requires a Hugging Face account and accepting the terms of ILSVRC/imagenet-1k)
hf auth login
hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/validation-*" --local-dir <D>/imagenet_hf

# 2. Convert it: <D>/imagenet_val/X (10 images per class) and <D>/imagenet_val/eval (the 40 000 others)
python prepare_imagenet_val.py <D>/imagenet_hf <D>/imagenet_val

# 3. Compute and evaluate
python demo.py -m resnet50 -t <D>/imagenet_val/X --cache <D>/cache/imagenet_val_X.pt
python evaluate.py -m resnet50 -v <D>/imagenet_val/eval --out results/imagenet_resnet50_10k

# Or, for several models in a row (results in results/imagenet_<model>_10k/)
DATA_DIR=<D> ./run_models.sh convnext_base swin_b vit_b_16 maxvit_t
```
The cached images are 224x224: choose models that take 224x224 inputs.

### Computing a universal perturbation for your model

To compute a universal perturbation for your model, please follow the same structure as in `demo.py`.
In particular, you should use the `universal_perturbation` function (see `universal_pert.py` for details), with the set of training images
used to compute the perturbation (tensor of size Mx3xHxW with values in [0, 1]), and your model in eval mode, which takes [0, 1] images as input
and outputs the logits (values before softmax).

Note: `data/universal.npy` is the perturbation precomputed with the former TensorFlow version, for the Inception model; it is not used by the PyTorch code.

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.

[2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
[*DeepFool: a simple and accurate method to fool deep neural networks*](http://arxiv.org/pdf/1511.04599), CVPR 2016.
