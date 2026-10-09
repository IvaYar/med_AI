# med_AI — Brain MRI Tumor Detection

med_AI is a team project developing a brain-MRI tumor detection workflow with
**Ultralytics YOLOv8n**. The intended model will locate tumors with bounding boxes
and assign one of three categories: `glioma`, `meningioma`, or `pituitary`.
Images from the `notumor` category serve as negative examples with no tumor boxes.

**Current stage: dataset preparation and initial configuration.** The local
workspace contains a prepared dataset and working preparation scripts. Model
training, predictions, and performance evaluation are planned next.

## What Is Available Now

| Component | Current status |
| --- | --- |
| Dataset download and source checks | Completed locally; source files preserved |
| Annotation validation and class mapping | Implemented in `prepare_dataset.py` |
| Train / validation / test split | Generated locally with seed `42` |
| Exact duplicate handling | Identical images kept within the same split |
| YOLO dataset configuration | `data.yaml` defines split paths and three classes |
| Python environment | Python 3.12 environment created and dependencies installed locally |
| Initial entry point | `main.py` prints settings and checks image/label pairs |
| Model training and prediction | Planned |
| Evaluation and result analysis | Planned; no model metrics reported yet |

### GitHub Snapshot and Local Workspace

The preparation source files in `main_project/` are included in this repository.
Cloning it provides the scripts, dependency list, and dataset configuration.
The dataset, prepared image/label copies, and Python environment are local files
and must be set up separately on each machine using the instructions below.

## Project Workspace

The following tree shows the implemented project files and generated local data:

```text
med_AI/
├── README.md                         # Project overview and team setup guide
└── main_project/
    ├── README.md                     # Detailed preparation notes
    ├── main.py                       # Initial settings and dataset checks
    ├── prepare_dataset.py            # Annotation preparation and split generation
    ├── requirements.txt              # Pinned Python dependencies
    ├── data.yaml                     # YOLO image paths and class names
    ├── .venv/                        # Local Python environment
    ├── dataset/
    │   ├── DATASET_SOURCE.json        # Local download provenance and source checks
    │   └── Brain Tumor labeled dataset/
    │       ├── glioma/
    │       ├── meningioma/
    │       ├── notumor/
    │       └── pituitary/
    └── data/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── preparation_report.json   # Counts, exclusions, and annotation issues
        └── split_manifest.json       # Source/output paths, hashes, and assignments
```

### What Each Script Does

`prepare_dataset.py` reads the original images and YOLO annotations, validates
annotation values, maps the positive class IDs, and creates the three dataset
splits. It copies images without modifying their bytes and writes prepared labels
into `data/`. The original files in `dataset/` remain unchanged. The script uses
the Python standard library and also generates the preparation report and manifest.

`main.py` imports PyTorch, Ultralytics, and PyYAML, applies the initial settings,
and checks that each split has image and label folders with a matching `.txt`
file for every image. Its current output is a configuration summary and image
counts. It does not instantiate the YOLO model or run training or predictions.

`data.yaml` supplies the image paths and class names for the future YOLO workflow.
The split paths are relative to `main_project/`.

## Dataset

The project uses version 1 of the public
[Labeled MRI brain Tumor dataset on Kaggle](https://www.kaggle.com/datasets/ammarahmed310/labeled-mri-brain-tumor-dataset),
listed as **CC0: Public Domain**. The downloaded source contains **2,175 JPEG
images** with paired text annotations across four source folders.

### Prepared Split

The preparation script uses approximately **70% training / 15% validation / 15%
testing**, with seed `42`. It assigns images by source category and keeps exact
image duplicates together. Group sizes and rounding can affect the proportions.

| Source category | Original images | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| glioma | 455 | 317 | 68 | 68 |
| meningioma | 550 | 385 | 83 | 82 |
| notumor | 550 | 385 | 83 | 82 |
| pituitary | 620 | 434 | 93 | 93 |
| **Total** | **2,175** | **1,521** | **327** | **325** |

The prepared dataset contains **2,173 image/label pairs**. Two `glioma` images,
`Tr-gl_0250.jpg` and `Tr-gl_0256.jpg`, have empty positive annotations and are
excluded. Training will use `train`; validation during development will use
`val`; the final model assessment will use `test`.

### Classes and Labels

| Detector class ID | Category | Original positive class ID |
| ---: | --- | ---: |
| 0 | glioma | 2 |
| 1 | meningioma | 1 |
| 2 | pituitary | 0 |

The mapping is based on the source folders and their consistent positive labels.
Prepared positive annotations preserve the original bounding-box coordinates.
The downloaded `classes.txt` files are not used as the class mapping.

All **550 `notumor` images** receive empty prepared `.txt` files. They represent
negative examples, so `notumor` is not a fourth detector class. Their downloaded
annotations contain boxes; those source files are preserved, while the copied
labels follow the project's no-tumor requirement. The orphan annotation
`meningioma/gg (9).txt` is not copied.

An image and its label share the same filename stem, for example:

```text
data/images/train/example.jpg
data/labels/train/example.txt
```

The label format is `class_id x_center y_center width height`, with coordinates
normalized to the image dimensions. This follows the
[Ultralytics YOLO detection dataset format](https://docs.ultralytics.com/datasets/detect/).
Empty text files are used for the prepared negative examples.

## Setup in VS Code

These instructions use **Windows PowerShell and Python 3.12**. A GPU is not
required for the current preparation and configuration checks.

### 1. Open the Workspace

Open the `med_AI` folder in VS Code and choose **Terminal → New Terminal**.
If you are setting up another machine, you can clone the repository with:

```powershell
git clone https://github.com/IvaYar/med_AI.git
cd med_AI
code .
```

From a terminal at the repository root, run:

```powershell
cd main_project
```

### 2. Create the Python Environment

Run this once per machine:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The direct dependencies are `ultralytics==8.4.174` and `PyYAML==6.0.3`.
Ultralytics installs its required dependencies, including PyTorch.
The commands use the environment's interpreter directly, so activating the
environment in PowerShell is optional.

For the VS Code Run button, open `Ctrl + Shift + P`, choose
**Python: Select Interpreter → Enter interpreter path**, and select:

```text
main_project/.venv/Scripts/python.exe
```

### 3. Place the Source Dataset

Download version 1 from the Kaggle dataset page and extract it into
`main_project/dataset/`. The preparation script expects this exact layout:

```text
main_project/dataset/Brain Tumor labeled dataset/glioma/
main_project/dataset/Brain Tumor labeled dataset/meningioma/
main_project/dataset/Brain Tumor labeled dataset/notumor/
main_project/dataset/Brain Tumor labeled dataset/pituitary/
```

Each source image must have its paired `.txt` file. Avoid adding an extra archive
folder between `dataset/` and `Brain Tumor labeled dataset/`.
On the existing development machine, the dataset is already in place.

### 4. Prepare Train, Validation, and Test Data

From `main_project/`, run:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

This generates `data/images/`, `data/labels/`, `data.yaml`, and the two JSON
reports. With the current source data, the split counts are:

```text
train: 1521 images
val: 327 images
test: 325 images
Prepared: 2173; excluded due to missing positive annotations: 2
```

Rerunning the script with the same source data and seed reproduces the assignments.
It verifies existing prepared files and stops if they have been changed, if
unexpected files appear in `data/`, or if an existing `data.yaml` differs from
the generated configuration. Review such a message before replacing any files.

### 5. Run the Initial Configuration Check

```powershell
.\.venv\Scripts\python.exe main.py
```

An example CPU run prints:

```text
Model: yolov8n.pt
Seed: 42; image size: 640; device: cpu
Dataset configuration: <workspace>/main_project/data.yaml
Classes: {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}
train: 1521 images, labels found
val: 327 images, labels found
test: 325 images, labels found
Initial settings are ready.
```

This confirms that the current configuration and image/label pairing checks pass.
It does not measure detection quality.

## Initial Settings

The following constants are defined near the top of `main_project/main.py`:

| Setting | Value | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `yolov8n.pt` | Selected YOLOv8 nano model for the planned detector |
| `SEED` | `42` | Initial random and PyTorch seed |
| `IMAGE_SIZE` | `640` | Intended image size for later model input |
| `DEVICE` | `cuda:0` when available, otherwise `cpu` | Device selected through PyTorch |
| `DATA_YAML` | `main_project/data.yaml` | Dataset configuration location |

`prepare_dataset.py` defines its own `SEED = 42` and the split ratios. Changing
the seed in `main.py` alone does not change the dataset assignments. Images are
not resized during preparation; `IMAGE_SIZE` is an initial setting for later use.
Model weights are not downloaded by the current entry point.

## Preparation Evidence and Open Issues

The local preparation checks have confirmed **2,173 image/label pairs**, unchanged
source files, valid positive annotation values, empty negative labels, and no
exact image duplicates shared between splits. All prepared JPEG files were also
checked for successful three-channel decoding.

Use these files to inspect the preparation results:

| File | Contents |
| --- | --- |
| `dataset/DATASET_SOURCE.json` | Local source version, license record, archive hash, and original-data checks |
| `data/preparation_report.json` | Split counts, class mapping, exclusions, source fingerprint, and duplicate annotation conflicts |
| `data/split_manifest.json` | One entry per prepared image, linking source paths, output paths, SHA-256, category, and split |

The current dataset still has issues to review before interpreting model results:

- There are **55 exact duplicate groups**. Each group stays within one split;
  the current allocation places all of them in training.
- **12 positive duplicate groups** have different bounding boxes for identical
  images. The paired labels are retained and flagged in the report for review.
- Patient identifiers are unavailable. This is an image split, so keeping exact
  copies together does not guarantee separation by patient or visually similar
  images.
- The source includes **353 grayscale JPEGs**. Preparation preserves image bytes;
  the standard YOLO three-channel loader handles grayscale images when loading.
- Class names come from the original folders. The images have not undergone an
  independent medical review.

The project is an educational prototype. Clinical validation is outside its
current scope, and no detection accuracy claims have been made.

## Working Together

Team members should use the same source dataset version, dependency versions,
class mapping, and seed to reproduce the current preparation results. Share
source code through GitHub; download the dataset separately
and recreate `.venv/` and prepared `data/` locally. Keep environments, dataset
archives, generated images, and future model weights out of source commits.

Before starting work on tracked files, update the repository from its root:

```powershell
git pull --ff-only
```

When contributing a change, describe what behavior changed and which command you
ran to verify it. If dataset preparation changes, include the resulting counts
and any new annotation issues in the review.

## Next Steps

1. Review the flagged annotation conflicts and agree on their treatment.
2. Add YOLOv8n model initialization and a reproducible training configuration.
3. Evaluate the trained model, then run predictions with visible bounding boxes.
4. Analyze errors and document results using the held-out test split.

Reference model documentation:
[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/).
