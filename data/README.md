# VPD-18

**Visual Perception Degradation: 18 patterns for autonomous driving**

[Google Drive](https://drive.google.com/drive/folders/REPLACE_WITH_VPD18_FOLDER_ID) · [Category table](categories.csv) · [Training guide](../docs/USAGE.md) · [Home](../README.md)

## Download

Download the image archive from **[Google Drive](https://drive.google.com/drive/folders/REPLACE_WITH_VPD18_FOLDER_ID)**. Store the images outside the Git history and prepare the class directories as described below.

## Categories

[![Original samples and multi-level features for the 18 VPD-18 degradation patterns](../assets/dataset_visualization.png)](../assets/dataset_visualization.pdf)

For each pattern, the figure shows the original degraded image and its low-, mid-, and high-level features. [View PDF](../assets/dataset_visualization.pdf).

The following statistics are reported in Table I of the paper.

| No. | Group | Degradation pattern | Images |
| --- | --- | --- | ---: |
| 1 | Contrast/Color | Color Temperature Shift | 6,798 |
| 2 | Contrast/Color | Contrast Enhancement | 6,798 |
| 3 | Contrast/Color | Histogram Equalization | 6,798 |
| 4 | Contrast/Color | Solarization | 6,798 |
| 5 | Contrast/Color | Saturation Distortion | 6,798 |
| 6 | Obstruction | Block Blotch Occlusion | 6,798 |
| 7 | Obstruction | Horizontal Stripe Occlusion | 6,798 |
| 8 | Obstruction | Vertical Stripe Occlusion | 6,798 |
| 9 | Obstruction | Lens Contamination | 6,798 |
| 10 | Noise | Monochrome Impulse Noise | 6,798 |
| 11 | Noise | Color Impulse Noise | 6,798 |
| 12 | Motion | Motion Blur | 6,798 |
| 13 | Abnormal Lighting | Low Illumination | 6,798 |
| 14 | Abnormal Lighting | Overexposure | 6,798 |
| 15 | Adverse Weather | Fog | 1,800 |
| 16 | Adverse Weather | Snow | 1,800 |
| 17 | Adverse Weather | Rain | 1,800 |
| 18 | Adverse Weather | Sandstorm | 1,800 |

**Total: 102,372 images.** The machine-readable table is available in [categories.csv](categories.csv).

Table numbers are paper category numbers. The model's class indices are assigned from alphabetically sorted class directory names; they are not determined by the order of this table.

## Training directory

The original data loader expects images directly inside each class directory:

```text
dataset/
├── <class_folder_1>/
│   ├── image_001.jpg
│   └── image_002.jpg
├── <class_folder_2>/
│   └── image_003.jpg
└── ...
```

The names above illustrate the format; retain the actual category names associated with your data and checkpoint. The loader accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, and `.webp`. It reads one directory level and does not recursively scan camera subfolders.

The training script makes a per-class 80/20 train/validation split with seed 42. It assigns labels using sorted immediate directory names. Preserve this class-to-index mapping when sharing or loading a trained model.

## Data preparation layout

The [directory reference](temporary_directory_structure.txt) records the supplied processing layout, including source images, software transformations, weather transformations, annotations, and camera views. It documents the preparation-stage hierarchy rather than the class-directory layout consumed by the training script.

For generated images, see the [degradation generation guide](../docs/USAGE.md#generating-degraded-images).
