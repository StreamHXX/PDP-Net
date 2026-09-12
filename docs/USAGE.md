# Training and data generation

## Environment

The paper reports Python 3.8, PyTorch 2.0.0, CUDA 11.8, Ubuntu 20.04, and an RTX 4090D. The requirements files list dependencies without version pins; use compatible package versions for your Python and CUDA environment.

The original implementation uses `torch.cuda.amp` and enables several CUDA performance settings when a GPU is available. Although it includes a CPU device fallback, a CPU training run has not been verified. The generation script includes `np.bool = np.bool_`; this patch alone is not a validated compatibility guarantee across NumPy/imgaug versions.

## Training and evaluation

Run from the repository root after preparing `dataset/<class_folder>/<image>`:

```bash
python "PDPNet_model.py"
```

The supplied script uses relative paths `./dataset`, `./pretrained/resnet50-11ad3fa6.pth`, and `./results_multibranch`. If the local backbone weights are absent or cannot be loaded, the original code requests torchvision's ImageNet-pretrained ResNet50 weights. A task-trained PDP-Net checkpoint is not included.

The original entry point handles training, selects the best checkpoint by validation macro F1, and evaluates it on the same validation split. It saves `best_model.pth`, `report.txt`, `training_history.png`, `confusion_matrix.png`, and `per_class_metrics.png` under `results_multibranch` on successful completion. It has no separate test-set or prediction command.

The script defaults are: image size 224, batch size 64, 10 epochs, learning rates `1e-5` (backbone group) and `1e-4` (other group), weight decay 0.01, auxiliary weights 0.3, and seed 42. See the source for the full configuration. The script defaults and the paper experiment settings are distinct; refer to the paper when reproducing its experiments.

The loader assigns labels from sorted class folder names. The original checkpoint saving code does not save this mapping, so record the actual directory-to-index mapping alongside any checkpoint you publish. The loader substitutes a gray image if an image fails to open; check dataset integrity before interpreting results.

## Generating degraded images

Install the separate generation dependencies in a compatible environment:

```bash
python -m pip install -r requirements-generation.txt
```

`main.py` retains its original machine-specific input/output paths. Its existing function can be called with explicit paths without editing the file. For example, run the following Python code from the repository root, replacing the example paths first:

```python
from main import batch_generation

batch_generation(
    image_path="/path/to/one/camera/image_folder",
    save_path="/path/to/a/new/output_folder",
    mode=3,
)
```

This invokes the original function without altering augmentation strengths or implementation. The input must contain readable images directly, not nested camera folders. The function reads the entire input directory into memory, so choose an input batch that fits available memory. Use a new output directory, because the original writer can replace files with matching names.

Existing mode groups:

| Mode | Operations in the source |
| --- | --- |
| 0 | Noise, block occlusion, hue/saturation variants, horizontal stripes, spatter, low light, overexposure, motion blur |
| 1 | Dominant color variants: red, orange, yellow, green, cyan, blue, purple, gray |
| 2 | Solarization, equalization, autocontrast, contrast adjustment, vertical stripes |
| 3 | Column fixed-pattern noise (`lambda_colfpn`) |

These groups are implementation options, not a verified one-to-one mapping to the paper's 18 classes. The supplied `main.py` does not contain the full weather-dataset generation pipeline. Its color handling, random behavior, and image loading have been preserved as supplied.
