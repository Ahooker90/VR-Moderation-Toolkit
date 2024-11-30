# VR Moderation Toolkit: Moderating Illicit Virtual Reality Games via Multimodal Pretrained Models

The **VR Moderation Toolkit** is a state-of-the-art system designed to detect and moderate explicit content in virtual reality (VR) environments. It leverages multimodal pretrained models like CLIP to address the unique challenges of moderating VR-specific content, including non-human avatars and in-game imagery.

---

## Key Features

- **Multimodal Moderation:** Combines image and text analysis for robust moderation.
- **VR-Specific Optimization:** Targets human and non-human avatars.
- **Enhanced Generalization:** Outperforms existing tools on in-game and external datasets.

---

## Datasets

### In-Game Dataset
Collected from **VRChat**, with both human and non-human avatars.

| Category   | Total | Unsafe | Safe |
|------------|-------|--------|------|
| Human      | 302   | 151    | 151  |
| Non-Human  | 284   | 142    | 142  |
| Combined   | 586   | 293    | 293  |

### In-The-Wild Dataset
Sourced from online platforms to test generalization.

| Category   | Total | Unsafe | Safe |
|------------|-------|--------|------|
| Combined   | 826   | 413    | 413  |

---

## Evaluation
The system is tested on multiple datasets to benchmark performance. Key evaluation highlights:
1. **Non-Human Data Points:** Demonstrates superior performance in detecting explicit content among non-human avatars.
2. **Human Data Points:** Matches or exceeds leading moderation tools for human-centric explicit content.
3. **Generalization Testing:** Outperforms other tools by up to 71% on user-generated VR content.


## Results

### In-Game Evaluation
Our system significantly outperforms current tools in detecting explicit content in VR environments.

| Tool              | Accuracy (A) | Precision (P) | Recall (R) | F1-Score |
|--------------------|--------------|---------------|------------|----------|
| NSFW.js           | 0.5          | 0.67          | 0.01       | 0.01     |
| AWS Rekognition   | 0.79         | 1.0           | 0.58       | 0.73     |
| Google SafeSearch | 0.57         | 1.0           | 0.15       | 0.26     |
| **VR Moderation Toolkit** | **0.92** | **0.91** | **0.94** | **0.92** |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahooker90/VR-Moderation-Toolkit.git
   cd VR-Moderation-Toolkit

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
