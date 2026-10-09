# YOLOv8n Project Preparation

At this stage, `prepare_dataset.py` prepares the dataset, while `main.py`
contains imports, initial settings, and folder checks. The settings at the
top of `main.py` specify the model `yolov8n.pt`, `SEED = 42`, image size `640`,
and automatic selection of CUDA when a compatible GPU is available or CPU.
The model is not instantiated yet, weights are not downloaded, and training
and predictions are not run.

## Running in the VS Code Terminal

Open the project folder and create a separate Python 3.12 environment.
Create `.venv` once:

```powershell
cd "C:\Users\jarom\OneDrive\Desktop\med_AI\main_project"
py -3.12 -m venv .venv
```

Install the dependencies in this environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare the dataset, then check the initial settings:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe main.py
```

`main.py` prints the model, seed, image size, device, classes, and image count
for each split. It also checks that the folders and a `.txt` file for every
image exist. An empty label file is valid for a negative example.

## File Locations

```text
main_project/
├── dataset/                  # original files
├── prepare_dataset.py        # dataset preparation and splitting
├── data.yaml                 # YOLO paths and class names
├── main.py                   # imports, settings, and path checks
├── requirements.txt          # Python dependencies
└── data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── preparation_report.json
    └── split_manifest.json
```

An image and its label file have the same name before the extension, for
example, `images/train/example.jpg` and `labels/train/example.txt`.

`data.yaml` specifies the paths `data/images/train`, `data/images/val`, and
`data/images/test` relative to the `main_project/` folder. The class names are:
`0: glioma`, `1: meningioma`, `2: pituitary`.

The dataset is split into approximately **70% / 15% / 15%**: `train` for future
training, `val` for validation during training, and `test` for final evaluation.
Seed `42` makes the result reproducible. The script preserves proportions
across the original class folders; identical images remain in the same split.
Duplicate groups and rounding may cause the proportions to differ slightly.
The original files in `dataset/` are preserved; prepared copies are stored
in `data/`. Images are copied without changing their bytes, including
grayscale images. The standard YOLO loader converts them to three channels
when reading them.

Negative examples in `notumor` receive empty `.txt` files; they are images
without tumor boxes, rather than a separate detector class. Two images in
`glioma` with empty labels are excluded from the prepared data. Preparation
details are recorded in `data/preparation_report.json`, and file assignments
are recorded in `data/split_manifest.json`.

To match these classes, preparation assumes that the original folders
correctly identify the tumor type. In copied label files, the original IDs
are remapped: `2 → 0` for `glioma`, `1 → 1` for `meningioma`, and `0 → 2` for
`pituitary`. Box coordinates are preserved. Empty negative labels are created
in the copies; the original labels are not modified.

There are 55 groups of identical images. In 43 `notumor` groups, the prepared
labels are identical and empty. In 12 tumor groups, the original boxes differ:
each copy keeps its corresponding label file, and the entire group is assigned
to one dataset split. This prevents identical images from appearing in
different splits, but the differing boxes require a separate label quality
review later; the script flags them in the report.

The original data has no patient identifiers. This is an image split rather
than a patient split: different images of one person may appear in different splits.

To use the Run button in VS Code, select the interpreter through
`Ctrl + Shift + P` → `Python: Select Interpreter` → `Enter interpreter path`:
`main_project/.venv/Scripts/python.exe` inside the repository folder.

File format: [Ultralytics YOLO datasets](https://docs.ultralytics.com/datasets/detect/).
Selected model: [YOLOv8n](https://docs.ultralytics.com/models/yolov8/).
