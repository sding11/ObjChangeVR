# ObjChangeVR

This repository accompanies the paper **“ObjChangeVR: Object State Change Reasoning from Continuous Egocentric Views in VR Environments”**.  
The code implements the complete ObjChangeVR pipeline, including **dataset construction**, **object change reasoning**, and **evaluation**.

Unlike traditional 3D-QA settings, ObjChangeVR operates on **continuous VR trajectories**.

---

## Outline

- `dataset_construction/`  
  Unity-based data export and QA benchmark generation.

- `method/`  
  Core ObjChangeVR pipeline and evaluation code.

---


## Dataset Organization

All code assumes the following **fixed dataset layout**:

```
dataset/
├── architecture/
│   ├── 1/
│   │   ├── compressed/
│   │   ├── disappear_object/
│   │   ├── path/
│   │   ├── screenshot/
│   │   │   ├── data.csv
│   │   │   ├── after/
│   │   │   ├── before/
│   │   │   └── compressed/
│   │   ├── groundtruth/
│   │   │   ├── data.csv
│   │   │   ├── before/
│   │   └── └── after/
│   ├── 2/
│   └── ...
├── fastfood/
├── market/
├── museum/
└── village/
```

- Each **numeric folder** represents a **single independent trajectory**.

---

## Dataset Construction

### Unity Export

Unity scripts are located in:

```
dataset_construction/export_unity/
```

#### Scripts

- `CameraPathRecorder.cs`  
  Records camera trajectories and screenshots.

- `CameraPathPlayer.cs`  
  Replays recorded camera trajectories.

- `FlyCamera.cs`  
  Provides free camera control (recording only).

#### Installation Rules

- `CameraPathRecorder.cs` and `CameraPathPlayer.cs` are **mutually exclusive**.  
  **Only one may be active per Unity scene/session.**

- `FlyCamera.cs` **must be attached together with** `CameraPathRecorder.cs`.

- `CameraPathPlayer.cs` does **not** require `FlyCamera.cs`.

---

### QA Benchmark Generation

Generate QA pairs using VLM:

```bash
python dataset_construction/generate_qa.py
```

**Input**
- Screenshot images from `groundtruth/after/` and `screenshot/after/`

**Output**
- `generated_QA.csv`

---

## ObjChangeVR Method Pipeline

The core pipeline is implemented in:

```
method/method.py
```

### Output

For each trajectory:
- `results.csv` containing:
  - `GeneratedAnswer`
  - `Sub_Answers`
  - `RetrievedIndices`

---

## Evaluation

Evaluation code is provided in:

```
method/evaluation.py
```

### Metrics

- Strict Exact Match (EM)
- EM@τ (τ = 0.8 by default)
- Class-level F1 score:
  - `disappeared`
  - `never`
  - `always been there`

### Output

- Excel summary file
- Group-level statistics:
  - Short scenes
  - Long scenes
  - Overall average

---

## Citation

If you use ObjChangeVR in academic work, please cite:

```bibtex
@inproceedings{ObjChangeVR,
  title={ObjChangeVR: Object State Change Reasoning from Continuous Egocentric Views in VR Environments},
  author={Ding, Shiyi and Wu, Shaoen and Chen, Ying},
  booktitle={Proceedings of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2026}
}
```
---
