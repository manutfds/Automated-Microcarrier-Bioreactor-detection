# Annotation Guidelines for Microcarrier Colonization Detection

This document provides detailed guidelines for annotating fluorescence microscopy images of microcarrier cultures. Consistent annotation is critical for training accurate detection models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Class Definitions](#2-class-definitions)
3. [Visual Examples](#3-visual-examples)
4. [Annotation Rules](#4-annotation-rules)
5. [Handling Edge Cases](#5-handling-edge-cases)
6. [Quality Control](#6-quality-control)
7. [Tools and Format](#7-tools-and-format)

---

## 1. Overview

### Purpose

Annotations serve as ground truth for training object detection models to:
- **Detect** individual microcarriers in images
- **Classify** each microcarrier as FULL (colonized) or EMPTY (non-colonized)

### Annotation Task

For each image:
1. Draw a bounding box around every visible microcarrier
2. Assign the appropriate class label (FULL or EMPTY)

---

## 2. Class Definitions

### Class 0: EMPTY

**Definition:** A microcarrier with NO visible cell attachment.

**Visual characteristics:**
- Dark, uniform appearance
- No internal green fluorescent signal
- May show slight autofluorescence at edges (still EMPTY)

### Class 1: FULL

**Definition:** A microcarrier with ANY degree of cell colonization.

**Visual characteristics:**
- Visible green fluorescent signal inside the microcarrier
- Signal can be partial, complete, bright, or dim
- Even minimal cell attachment = FULL

### Key Principle

> **Any visible fluorescence inside the microcarrier = FULL**
>
> The distinction is binary: either cells are present (FULL) or not (EMPTY).
> Do not consider the degree of colonization for classification.

---

## 3. Visual Examples

### EMPTY Microcarriers

```
Characteristics:
┌─────────────────────────────────────────────────────────┐
│  • Dark, spherical appearance                           │
│  • No internal green signal                             │
│  • Uniform gray/black interior                          │
│  • Edge autofluorescence is acceptable (still EMPTY)    │
└─────────────────────────────────────────────────────────┘
```

**Examples of EMPTY:**
- Completely dark microcarrier
- Microcarrier with slight edge glow but no internal signal
- Microcarrier that appears "hollow"

### FULL Microcarriers

```
Characteristics:
┌─────────────────────────────────────────────────────────┐
│  • Visible green fluorescent signal inside              │
│  • Can be bright OR dim                                 │
│  • Can be partial OR complete coverage                  │
│  • Any internal fluorescence = FULL                     │
└─────────────────────────────────────────────────────────┘
```

**Examples of FULL:**
- Brightly fluorescent microcarrier (high cell density)
- Partially covered microcarrier (cells on one side)
- Dimly fluorescent microcarrier (few cells)
- Microcarrier with scattered fluorescent spots

---

## 4. Annotation Rules

### Rule 1: Include Only Fully Visible Microcarriers

✅ **Annotate:**
- Microcarriers that are completely within the image frame
- Microcarriers where the entire boundary is visible

❌ **Do NOT annotate:**
- Microcarriers cropped at image borders
- Microcarriers where part of the boundary is outside the frame

### Rule 2: One Box Per Microcarrier

- Each microcarrier gets exactly ONE bounding box
- Do not create multiple boxes for the same object
- Do not merge multiple microcarriers into one box

### Rule 3: Tight Bounding Boxes

- Draw boxes that closely fit the microcarrier boundary
- Minimize empty space around the object
- Include the entire microcarrier within the box

### Rule 4: Overlapping Microcarriers

- Annotate each microcarrier separately even if they overlap
- Bounding boxes CAN overlap
- If microcarriers are so close they appear merged, use best judgment to define separate boundaries

### Rule 5: Binary Classification Only

- Choose EMPTY or FULL — no intermediate categories
- When in doubt, refer to the principle: "Any visible internal fluorescence = FULL"

---

## 5. Handling Edge Cases

### Case 1: Ambiguous Fluorescence

**Situation:** Very faint internal signal, unclear if it's cell fluorescence or artifact.

**Guideline:** 
- If you can distinguish it from background noise → FULL
- If indistinguishable from background → EMPTY
- When truly uncertain → Mark for consensus review

### Case 2: Debris Inside Microcarrier

**Situation:** Non-cell material appears inside the microcarrier.

**Guideline:**
- Debris typically appears as irregular, non-uniform shapes
- Cell fluorescence is more uniform/diffuse
- If clearly debris (not cells) → EMPTY

### Case 3: Out-of-Focus Microcarriers

**Situation:** Microcarrier is blurry due to focal plane.

**Guideline:**
- If class can still be determined → Annotate normally
- If completely uninterpretable → Do not annotate (exclude)

### Case 4: Touching Microcarriers

**Situation:** Two or more microcarriers are in contact.

**Guideline:**
- Draw separate boxes for each
- Boxes may overlap
- Classify each independently

### Case 5: Very Small Objects

**Situation:** Object appears much smaller than typical microcarriers.

**Guideline:**
- If clearly a microcarrier (correct morphology) → Annotate
- If debris or artifact → Do not annotate
- Typical microcarriers have consistent, circular morphology

### Case 6: Artifacts and Debris

**Situation:** Non-microcarrier objects in the image.

**Guideline:**
- Do NOT annotate debris, bubbles, or other artifacts
- Only annotate objects with clear microcarrier morphology
- Microcarriers are circular/spherical; debris has irregular shapes

---

## 6. Quality Control

### Self-Check Checklist

Before completing each image, verify:

- [ ] All fully visible microcarriers are annotated
- [ ] No cropped microcarriers are annotated
- [ ] Each microcarrier has exactly one box
- [ ] Boxes are tightly fit
- [ ] Class labels are correct (FULL vs EMPTY)
- [ ] No obvious debris/artifacts are annotated

### Consensus Review Process

For ambiguous cases:

1. Mark the image for review
2. Multiple annotators independently classify
3. Discuss disagreements as a group
4. Reach consensus decision
5. Document the reasoning

### Inter-Annotator Agreement

To ensure consistency:

- Calculate agreement rate on subset of images
- Target: >95% agreement on class labels
- Review and discuss systematic disagreements

---

## 7. Tools and Format

### Recommended Annotation Tool

**LabelImg** (version 1.8.6 or later)
- Free and open source
- Supports YOLO format export
- Easy to use

**Alternatives:**
- CVAT (web-based)
- Roboflow (cloud-based)
- Label Studio (flexible)

### Output Format: YOLO

Each image `image_name.jpg` has a corresponding `image_name.txt` with annotations:

```
<class_id> <x_center> <y_center> <width> <height>
```

**All values are normalized (0-1):**
- `class_id`: 0 for EMPTY, 1 for FULL
- `x_center`: Center X coordinate / Image width
- `y_center`: Center Y coordinate / Image height
- `width`: Box width / Image width
- `height`: Box height / Image height

**Example annotation file:**
```
0 0.4521 0.3125 0.0856 0.0923
1 0.7234 0.5678 0.0912 0.0887
0 0.1567 0.8234 0.0834 0.0901
1 0.6123 0.2345 0.0878 0.0912
```

### File Organization

```
annotations/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.txt
    ├── img002.txt
    └── ...
```

---

## Summary

| Aspect | Guideline |
|--------|-----------|
| **EMPTY** | No internal fluorescence |
| **FULL** | Any internal fluorescence |
| **Border objects** | Exclude (do not annotate) |
| **Overlapping** | Annotate separately |
| **Debris** | Exclude (do not annotate) |
| **Ambiguous** | Mark for consensus review |
| **Format** | YOLO (normalized coordinates) |

---

*For questions, contact: manu1torres@gmail.com*
