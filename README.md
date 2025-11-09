# FlyAI — Prompt-to-Aircraft + CFD Optimizer

FlyAI turns a text prompt into an aircraft concept, runs CFD, and iterates to reduce drag.

> **Quick start:**  
> **You can run everything by executing:**  
> `python all2/main.py`

---

## Videos:
https://sebastian.microflux.de/demo.mp4
https://sebastian.microflux.de/tech.mp4

## Prereqs
- Python 3.10+ (Conda/Mamba recommended)
- (Optional but recommended) System deps used by the pipeline: CPACS/TiGL, SU2, OpenMPI, and a working OpenGL/OSMesa stack for off-screen renders.

If you already have a conda env set up, you’re good. Otherwise:

```bash
# option A: from YAML if available
conda env create -f environment.full.yml
conda activate flyai

# option B: minimal
conda create -n flyai python=3.11 -y
conda activate flyai
pip install -r requirements.txt
