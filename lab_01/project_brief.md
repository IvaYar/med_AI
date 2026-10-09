# Project Development Practicum — Project Brief

Read `README.md` before editing this file. Replace only the prompts in square
brackets, including the brackets. Keep every uppercase field name and colon.
An answer may continue on the following line.

Use the same member labels in later labs. A member label is a first name or
short initials.

## Team

TEAM-NAME: MRI-Team-WYK
MEMBER-LABELS: Wan, Yaromir, Kareem
TEAM-WORK: rotate commands- Wan, review decisions- Yaromir, report problems that stop progress- Kareem.

## Fixed Course Facts

- The project is a brain-MRI object-detection workflow.
- The input is one RGB brain-MRI image.
- The provided dataset annotates regions as `glioma`, `meningioma`, or
  `pituitary`. A `notumor` image is a valid negative example with no boxes.
- The result is zero or more labeled boxes for human review.
- This is non-clinical coursework and not a diagnosis.
- The full dataset is introduced in Lab 04. Do not download it for Lab 01.

These are supplied facts, not questions. Do not rewrite them.

## Project Boundary

SYSTEM-DOES: The system receives brain-tumor MRI medical images, detects and classifies glioma, meningioma, and pituitary tumor regions, and outputs suspicious tumor bounding boxes as an auxiliary reference for radiologists.
SYSTEM-DOES-NOT: The system does not replace professional radiological diagnosis and can only act as an auxiliary screening tool.
REVIEWER: Radiologists receive and review suspicious tumor bounding boxes as an auxiliary reference for subsequent clinical diagnosis.

## Risk And Testable Behaviour

RISK: The detector may produce false bounding boxes on normal brain structures, such as blood vessels and cerebral gyri, because of massive no tumor negative samples.
RESPONSE: The project suppresses false positives through negative-sample sampling and inference-stage threshold adjustment and assesses the false-positive quantity.
TESTABLE-BEHAVIOUR: Given the no tumor negative samples, When controlled experiments with threshold adjustment are conducted, Then performance is assessed using mAP@0.5, Precision, Recall, F1-score, and false-positive quantity.

## Initial Contributions

INITIAL-CONTRIBUTIONS:
