#!/usr/bin/env python3
"""
setup_week1.py
==============
Week 1 environment bootstrapper for the Cambodian ALPR project.

What it does
------------
1. Verifies your Python version (>= 3.9).
2. Creates a virtual environment (.venv) in the project root.
3. Installs PyTorch with GPU (CUDA) support into that venv.
4. Installs YOLOv10 (Ultralytics fork) + clones the reference repos
   for YOLOv10 and CRNN into third_party/.
5. Installs OpenCV, numpy, pandas, roboflow and other core deps.
6. Verifies the GPU is visible from inside the venv
   (torch.cuda.is_available()).
7. Creates the full project directory structure.

Usage
-----
    # GPU machine (CUDA 12.1 is the default):
    python setup_week1.py

    # Choose a CUDA build explicitly:
    python setup_week1.py --cuda 118
    python setup_week1.py --cuda 121

    # No NVIDIA GPU (CPU-only build of PyTorch):
    python setup_week1.py --cuda cpu

    # Skip venv creation and install into the current interpreter:
    python setup_week1.py --no-venv

    # Only create folders / only verify GPU:
    python setup_week1.py --dirs-only
    python setup_week1.py --verify-only

Notes
-----
* The script never runs interactive prompts; it prints every command
  it runs so you can audit / re-run steps manually if needed.
* Re-running is safe: venv creation, pip installs, git clones and
  directory creation are all idempotent.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
THIRD_PARTY = PROJECT_ROOT / "third_party"

MIN_PYTHON = (3, 9)

# PyTorch install matrix. Keys are the --cuda choices.
TORCH_INDEX = {
    "118": "https://download.pytorch.org/whl/cu118",
    "121": "https://download.pytorch.org/whl/cu121",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# torch/vision are installed together from the CUDA index above.
TORCH_PACKAGES = ["torch>=2.0", "torchvision", "torchaudio"]

# Everything else comes from the normal PyPI index.
CORE_PACKAGES = [
    "ultralytics>=8.1.0",   # ships YOLOv10 model support (yolov10n/s/m/b/l/x)
    "opencv-python>=4.8",
    "numpy",
    "pandas",
    "roboflow",
    "albumentations",       # augmentation (used from week 2 onward)
    "matplotlib",
    "pyyaml",
    "tqdm",
    "pillow",
    "lmdb",                 # CRNN reference repo uses LMDB datasets
]

# Reference repositories cloned into third_party/ for inspection & training.
GIT_REPOS = {
    "yolov10": "https://github.com/THU-MIG/yolov10.git",
    "crnn.pytorch": "https://github.com/meijieru/crnn.pytorch.git",
}

# Project folder layout created under PROJECT_ROOT.
DIRECTORIES = [
    "data/raw",
    "data/raw/by_angle/front",
    "data/raw/by_angle/angled_left",
    "data/raw/by_angle/angled_right",
    "data/raw/by_angle/rear",
    "data/raw/by_lighting/daylight",
    "data/raw/by_lighting/low_light",
    "data/raw/by_lighting/backlit",
    "data/interim",              # renamed / cleaned images
    "data/annotated",            # Roboflow YOLO export lands here
    "data/metadata",             # CSV logs
    "models/detection",          # YOLOv10 weights
    "models/recognition",        # CRNN weights
    "models/pretrained",
    "src/detection",
    "src/recognition",
    "src/data",
    "src/utils",
    "configs",
    "notebooks",
    "scripts",
    "third_party",
    "docs",
    "logs",
    "outputs",
]

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

C_OK = "\033[92m"
C_WARN = "\033[93m"
C_ERR = "\033[91m"
C_INFO = "\033[96m"
C_END = "\033[0m"


def log(msg: str, color: str = C_INFO) -> None:
    print(f"{color}{msg}{C_END}")


def step(n: int, msg: str) -> None:
    print()
    log(f"[{n}] {msg}", C_INFO)
    print("-" * 70)


def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess, echoing the command first. Raises on failure."""
    printable = " ".join(str(c) for c in cmd)
    log(f"$ {printable}", C_WARN)
    subprocess.run(cmd, check=True, **kwargs)


def venv_python() -> Path:
    """Path to the python executable inside the created venv."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def check_python() -> None:
    step(1, "Checking Python version")
    if sys.version_info < MIN_PYTHON:
        log(
            f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"found {sys.version.split()[0]}",
            C_ERR,
        )
        sys.exit(1)
    log(f"OK: Python {sys.version.split()[0]}", C_OK)


def create_venv(use_venv: bool) -> Path:
    """Create the venv (if requested) and return the python to install into."""
    if not use_venv:
        log("Skipping venv creation (--no-venv). Using current interpreter.",
            C_WARN)
        return Path(sys.executable)

    step(2, "Creating virtual environment (.venv)")
    if VENV_DIR.exists():
        log(f"venv already exists at {VENV_DIR} — reusing it.", C_OK)
    else:
        builder = venv.EnvBuilder(with_pip=True, upgrade_deps=True)
        builder.create(str(VENV_DIR))
        log(f"Created venv at {VENV_DIR}", C_OK)

    py = venv_python()
    if not py.exists():
        log(f"ERROR: expected venv python not found at {py}", C_ERR)
        sys.exit(1)

    # Make sure pip is current.
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    return py


def install_pytorch(py: Path, cuda: str) -> None:
    step(3, f"Installing PyTorch (build: {cuda})")
    index = TORCH_INDEX[cuda]
    run([str(py), "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", index])
    log("PyTorch installed.", C_OK)


def install_core(py: Path) -> None:
    step(4, "Installing YOLOv10 / CRNN deps and core libraries")
    run([str(py), "-m", "pip", "install", *CORE_PACKAGES])
    log("Core packages installed.", C_OK)


def clone_repos() -> None:
    step(5, "Cloning reference repositories into third_party/")
    if shutil.which("git") is None:
        log("WARNING: git not found on PATH — skipping repo clones. "
            "Install git and re-run, or download the repos manually.", C_WARN)
        return
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    for name, url in GIT_REPOS.items():
        dest = THIRD_PARTY / name
        if dest.exists():
            log(f"{name} already cloned — skipping.", C_OK)
            continue
        run(["git", "clone", "--depth", "1", url, str(dest)])
    log("Reference repos ready.", C_OK)


def create_dirs() -> None:
    step(6, "Creating project directory structure")
    for rel in DIRECTORIES:
        d = PROJECT_ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
        # Keep empty dirs in version control.
        gitkeep = d / ".gitkeep"
        if not any(d.iterdir()):
            gitkeep.touch()
    log(f"Created {len(DIRECTORIES)} directories under {PROJECT_ROOT}", C_OK)


def verify_gpu(py: Path) -> None:
    step(7, "Verifying GPU availability (torch.cuda.is_available())")
    check = (
        "import torch;"
        "avail = torch.cuda.is_available();"
        "print('torch version   :', torch.__version__);"
        "print('CUDA available  :', avail);"
        "print('CUDA version    :', torch.version.cuda);"
        "print('device count    :', torch.cuda.device_count());"
        "print('device name     :', torch.cuda.get_device_name(0) if avail else 'N/A')"
    )
    result = subprocess.run([str(py), "-c", check])
    if result.returncode != 0:
        log("Could not import torch — check the install log above.", C_ERR)
        return
    log(
        "If 'CUDA available' is False on an NVIDIA machine, verify your "
        "GPU driver and that you installed a cuXXX (not cpu) build.",
        C_WARN,
    )


def print_next_steps(py: Path, use_venv: bool) -> None:
    activate = (
        f"{VENV_DIR}\\Scripts\\activate"
        if os.name == "nt"
        else f"source {VENV_DIR}/bin/activate"
    )
    print()
    log("=" * 70, C_OK)
    log("Week 1 environment setup complete.", C_OK)
    log("=" * 70, C_OK)
    if use_venv:
        print(f"\nActivate the environment with:\n    {activate}\n")
    print("Next actions (manual):")
    print("  1. Read DATA_COLLECTION_GUIDE.md")
    print("  2. Collect 100 Cambodian license plate photos (smartphone).")
    print("  3. Log each photo in data/metadata/metadata_log.csv")
    print("  4. Follow roboflow_setup.py to upload, annotate and export.")
    print("  5. Place the YOLO-format export in data/annotated/\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 1 ALPR environment setup.")
    parser.add_argument(
        "--cuda",
        choices=list(TORCH_INDEX.keys()),
        default="121",
        help="PyTorch build: 118 (CUDA 11.8), 121 (CUDA 12.1) or cpu. Default: 121.",
    )
    parser.add_argument("--no-venv", dest="use_venv", action="store_false",
                        help="Install into the current interpreter, not a new venv.")
    parser.add_argument("--dirs-only", action="store_true",
                        help="Only create the directory structure and exit.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run the GPU verification and exit.")
    parser.add_argument("--skip-clone", action="store_true",
                        help="Do not clone the YOLOv10 / CRNN reference repos.")
    args = parser.parse_args()

    log("Cambodian ALPR — Week 1 Setup", C_INFO)
    log(f"Project root: {PROJECT_ROOT}", C_INFO)

    if args.dirs_only:
        create_dirs()
        return

    if args.verify_only:
        py = venv_python() if VENV_DIR.exists() else Path(sys.executable)
        verify_gpu(py)
        return

    check_python()
    py = create_venv(args.use_venv)
    install_pytorch(py, args.cuda)
    install_core(py)
    if not args.skip_clone:
        clone_repos()
    create_dirs()
    verify_gpu(py)
    print_next_steps(py, args.use_venv)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        log(f"\nA command failed (exit {exc.returncode}). See output above.", C_ERR)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        log("\nInterrupted by user.", C_WARN)
        sys.exit(130)
