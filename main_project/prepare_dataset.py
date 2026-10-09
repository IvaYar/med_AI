"""Prepare train/val/test splits for YOLOv8n while preserving the source dataset."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "dataset" / "Brain Tumor labeled dataset"
OUTPUT_DIR = PROJECT_DIR / "data"
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
# Class mapping is based on the source folders and consistent positive labels.
SOURCE_CLASS_IDS = {"glioma": 2, "meningioma": 1, "pituitary": 0}
CLASS_IDS = {"glioma": 0, "meningioma": 1, "pituitary": 2}
CATEGORIES = ("glioma", "meningioma", "notumor", "pituitary")
DATA_YAML = """# Paths are relative to this file; path is intentionally omitted.
train: data/images/train
val: data/images/val
test: data/images/test
names:
  0: glioma
  1: meningioma
  2: pituitary
"""


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in SOURCE_DIR.rglob("*") if p.is_file()):
        entry = "{}:{}\n".format(path.relative_to(SOURCE_DIR).as_posix(), file_hash(path))
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


def read_boxes(label_path: Path) -> list[list[str]]:
    boxes = []
    for number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("{}:{}: five YOLO fields are required.".format(label_path.name, number))
        values = [float(value) for value in fields]
        class_id, x, y, width, height = values
        if not all(math.isfinite(value) for value in values):
            raise ValueError("{}:{}: all values must be finite numbers.".format(label_path.name, number))
        if not class_id.is_integer() or not 0 <= class_id <= 3:
            raise ValueError("{}:{}: invalid source class_id.".format(label_path.name, number))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError("{}:{}: coordinates are outside the 0-1 range.".format(label_path.name, number))
        if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
            raise ValueError("{}:{}: bounding box extends beyond the image boundaries.".format(label_path.name, number))
        boxes.append(fields)
    return boxes


def collect_samples() -> tuple[list[dict], list[dict]]:
    samples, excluded = [], []
    for category in CATEGORIES:
        folder = SOURCE_DIR / category
        if not folder.is_dir():
            raise FileNotFoundError("Source folder not found: {}".format(folder))
        for image in sorted(folder.glob("*.jpg")):
            label = image.with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError("Annotation file not found: {}".format(label))
            boxes = read_boxes(label)
            if category != "notumor" and not boxes:
                excluded.append({"source_image": image.relative_to(SOURCE_DIR).as_posix(), "reason": "missing_positive_annotation"})
                continue
            if category != "notumor" and any(int(float(box[0])) != SOURCE_CLASS_IDS[category] for box in boxes):
                raise ValueError("{}: class_id does not match the expected source category.".format(label))
            # The course treats notumor images as negative examples. Keep the source files unchanged.
            prepared_label = "" if category == "notumor" else "".join(
                "{} {}\n".format(CLASS_IDS[category], " ".join(box[1:])) for box in boxes
            )
            samples.append({"image": image, "label": label, "category": category,
                            "sha256": file_hash(image), "prepared_label": prepared_label})
    if not samples:
        raise ValueError("The source dataset contains no suitable images.")
    return samples, excluded


def allocate_counts(total: int) -> dict[str, int]:
    exact = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: math.floor(value) for split, value in exact.items()}
    order = sorted(counts, key=lambda split: exact[split] - counts[split], reverse=True)
    for split in order[:total - sum(counts.values())]:
        counts[split] += 1
    return counts


def assign_splits(samples: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["sha256"]].append(sample)
    if any(len({sample["category"] for sample in group}) != 1 for group in groups.values()):
        raise ValueError("Identical images have different categories; a review is required.")
    conflicts = []
    for category in CATEGORIES:
        category_groups = [group for group in groups.values() if group[0]["category"] == category]
        remaining = allocate_counts(sum(len(group) for group in category_groups))
        rng.shuffle(category_groups)
        category_groups.sort(key=len, reverse=True)
        for group in category_groups:
            candidates = [split for split in remaining if remaining[split] >= len(group)] or list(remaining)
            split = max(candidates, key=lambda name: remaining[name])
            remaining[split] -= len(group)
            for sample in group:
                sample["split"] = split
            annotations = {tuple(sorted(tuple(float(value) for value in line.split())
                                       for line in sample["prepared_label"].splitlines())) for sample in group}
            if len(annotations) > 1:
                conflicts.append({"sha256": group[0]["sha256"], "split": split,
                                  "source_images": [sample["image"].relative_to(SOURCE_DIR).as_posix() for sample in group]})
    return conflicts


def write_json(path: Path, value: dict | list) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit("Downloaded dataset not found: {}".format(SOURCE_DIR))
    original_hash = source_fingerprint()
    samples, excluded = collect_samples()
    conflicts = assign_splits(samples)
    manifest, expected_files = [], set()
    for sample in samples:
        name = "{}__{}".format(sample["category"], sample["image"].stem)
        sample["output_image"] = OUTPUT_DIR / "images" / sample["split"] / (name + ".jpg")
        sample["output_label"] = OUTPUT_DIR / "labels" / sample["split"] / (name + ".txt")
        expected_files.update((sample["output_image"], sample["output_label"]))
        manifest.append({"source_image": sample["image"].relative_to(PROJECT_DIR).as_posix(),
                         "source_label": sample["label"].relative_to(PROJECT_DIR).as_posix(),
                         "image_sha256": sample["sha256"], "category": sample["category"],
                         "split": sample["split"],
                         "image": sample["output_image"].relative_to(PROJECT_DIR).as_posix(),
                         "label": sample["output_label"].relative_to(PROJECT_DIR).as_posix()})
    if len(expected_files) != 2 * len(samples):
        raise ValueError("Duplicate output filenames detected.")
    report_path, manifest_path = OUTPUT_DIR / "preparation_report.json", OUTPUT_DIR / "split_manifest.json"
    if OUTPUT_DIR.exists():
        unexpected = {p for p in OUTPUT_DIR.rglob("*") if p.is_file()} - expected_files - {report_path, manifest_path}
        if unexpected:
            raise FileExistsError("Unexpected files found in data/. Preparation stopped without overwriting files.")
    config_path = PROJECT_DIR / "data.yaml"
    if config_path.exists() and config_path.read_bytes() != DATA_YAML.encode("utf-8"):
        raise FileExistsError("data.yaml already contains a different configuration; it has been preserved.")
    for sample in samples:
        image, label = sample["output_image"], sample["output_label"]
        if image.exists() and file_hash(image) != sample["sha256"]:
            raise FileExistsError("Prepared image has been modified: {}".format(image))
        if label.exists() and label.read_bytes() != sample["prepared_label"].encode("utf-8"):
            raise FileExistsError("Prepared annotation has been modified: {}".format(label))
    for sample in samples:
        image, label = sample["output_image"], sample["output_label"]
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists():
            shutil.copy2(sample["image"], image)
        if not label.exists():
            label.write_bytes(sample["prepared_label"].encode("utf-8"))
        if file_hash(image) != sample["sha256"]:
            raise RuntimeError("Image copy verification failed: {}".format(image))
    if source_fingerprint() != original_hash:
        raise RuntimeError("The source dataset changed during preparation.")
    split_hashes = {split: {s["sha256"] for s in samples if s["split"] == split} for split in SPLIT_RATIOS}
    if any(split_hashes[a] & split_hashes[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Exact image duplicates were assigned to different splits.")
    counts = {split: dict(Counter(s["category"] for s in samples if s["split"] == split)) for split in SPLIT_RATIOS}
    orphan_labels = [p.relative_to(SOURCE_DIR).as_posix() for p in sorted(SOURCE_DIR.rglob("*.txt"))
                     if p.name != "classes.txt" and not p.with_suffix(".jpg").exists()]
    report = {"seed": SEED, "ratios": SPLIT_RATIOS, "class_ids": CLASS_IDS,
              "source_class_ids": SOURCE_CLASS_IDS, "source_fingerprint_sha256": original_hash,
              "prepared_images": len(samples), "split_counts": counts, "excluded": excluded,
              "orphan_labels_not_copied": orphan_labels,
              "negative_annotations_cleared": sum(s["category"] == "notumor" for s in samples),
              "duplicate_groups": sum(count > 1 for count in Counter(s["sha256"] for s in samples).values()),
              "duplicate_groups_with_different_annotations": conflicts,
              "notes": ["Classes are based on the source folders; the images have not undergone a medical review.",
                        "notumor images are treated as negative examples under the project requirements.",
                        "Different bounding boxes for exact image duplicates have been preserved; the annotations need a later review.",
                        "The splits are not grouped by patient because patient IDs are unavailable.",
                        "Images are copied byte for byte; the YOLO three-channel loader handles grayscale images when reading them."]}
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    config_path.write_bytes(DATA_YAML.encode("utf-8"))
    for split, categories in counts.items():
        print("{}: {} images; {}".format(split, sum(categories.values()), categories))
    print("Prepared: {}; excluded due to missing positive annotations: {}".format(len(samples), len(excluded)))
    print("Groups with different bounding boxes: {}; see data/preparation_report.json".format(len(conflicts)))
    print("Source files have been preserved; no exact image duplicates occur across splits.")


if __name__ == "__main__":
    main()
