#!/usr/bin/env python3
"""
roboflow_setup.py
=================
Roboflow integration helper for the Cambodian ALPR project (Week 1).

Roboflow is where you will UPLOAD the 100 photos, DRAW bounding boxes around
each license plate (annotation), and EXPORT the labelled data in YOLO format
for YOLOv10 training.

This script has two modes:

  1. GUIDE MODE (default) — prints the full step-by-step instructions for
     creating an account, uploading, annotating and exporting. No account or
     API key needed. Just run it and read.

         python roboflow_setup.py

  2. DOWNLOAD MODE — once your dataset is annotated and a "version" has been
     generated on Roboflow, this pulls the YOLO-format export straight into
     data/annotated/ using the Roboflow Python SDK.

         python roboflow_setup.py --download \
             --api-key   YOUR_KEY \
             --workspace your-workspace \
             --project   cambodian-alpr \
             --version   1

     (You can also set the env var ROBOFLOW_API_KEY instead of --api-key.)

Free tier is enough for Week 1: https://roboflow.com  → "Public" project.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Single detection class for Week 1. (CRNN text-reading comes later.)
CLASS_NAME = "license_plate"


# --------------------------------------------------------------------------- #
# GUIDE MODE
# --------------------------------------------------------------------------- #

GUIDE = f"""
======================================================================
 ROBOFLOW SETUP GUIDE — Cambodian ALPR (Week 1)
======================================================================

You will: create an account -> new project -> upload 100 images ->
draw one box per plate -> generate a version -> export in YOLO format.

----------------------------------------------------------------------
STEP 1 — Create a free Roboflow account
----------------------------------------------------------------------
  1. Go to https://roboflow.com and click "Sign Up".
  2. Sign up (Google login is fine). Free tier is enough for Week 1.
  3. When asked, create a WORKSPACE (e.g. "alpr-cambodia").
     Note the workspace URL slug — you'll need it to download later.

----------------------------------------------------------------------
STEP 2 — Create a new project
----------------------------------------------------------------------
  1. Click "Create New Project".
  2. Project Name : cambodian-alpr
  3. Project Type : "Object Detection"     <-- important (bounding boxes)
  4. Annotation Group / class name: "{CLASS_NAME}"
  5. License: your choice (CC BY 4.0 is fine for a student project).

----------------------------------------------------------------------
STEP 3 — Upload your 100 images
----------------------------------------------------------------------
  1. Open the project -> "Upload" tab.
  2. Drag the entire folder:
         {RAW_DIR}
     (or select all 100 renamed .jpg files).
  3. Wait for the thumbnails to load, then click "Save and Continue".
  4. Roboflow asks how to split the data. For now accept the default
     TRAIN / VALID / TEST split (e.g. 70 / 20 / 10) — you can change it
     when you generate the version.

  Tips:
   * Convert any HEIC files to JPG BEFORE uploading.
   * If images are huge, Roboflow auto-resizes on export (set in Step 6).

----------------------------------------------------------------------
STEP 4 — Annotate: draw a bounding box around every plate
----------------------------------------------------------------------
  1. Go to the "Annotate" tab and open the first image.
  2. Pick the Bounding Box tool (keyboard: 'b').
  3. Draw a TIGHT rectangle around the license plate:
        - Include the whole plate (all characters + border).
        - Do NOT include the bumper / lots of background.
        - If a photo has two visible plates, box BOTH.
  4. Assign the class "{CLASS_NAME}".
  5. Press the RIGHT ARROW to go to the next image. Repeat for all 100.

  Annotation quality rules (this decides model accuracy):
   * One box per visible plate, tight but not clipping characters.
   * Be consistent: same tightness on every image.
   * Multi-line Khmer plates -> ONE box around the whole plate block.
   * QR-code plates -> box the whole plate INCLUDING the QR area.
   * If a plate is unreadable/too blurry, either skip the image or
     mark it "null" — don't draw a guess box.

----------------------------------------------------------------------
STEP 5 — (Optional) Preprocessing & Augmentation
----------------------------------------------------------------------
  When you click "Generate", Roboflow offers preprocessing + augmentation.
  For the Week 1 baseline keep it SIMPLE:
     Preprocessing:
        - Auto-Orient : ON
        - Resize      : 640 x 640 (stretch or fit) -- matches YOLOv10 input
     Augmentation:
        - Leave OFF for the first version (you want a clean baseline).
          You'll add augmentation from Week 2 onward.

----------------------------------------------------------------------
STEP 6 — Generate a dataset version
----------------------------------------------------------------------
  1. Go to "Generate" -> confirm the train/valid/test split.
  2. Apply the preprocessing from Step 5.
  3. Click "Generate". This creates VERSION 1 of the dataset.
     -> Remember the version NUMBER (1). You need it to download.

----------------------------------------------------------------------
STEP 7 — Export in YOLO format
----------------------------------------------------------------------
  Two ways:

  (A) Manual download:
      1. Open the generated version -> "Export Dataset".
      2. Format: choose "YOLOv8" (YOLOv10 uses the same label format:
         one .txt per image, "class cx cy w h" normalised 0-1).
      3. Choose "Download zip to computer".
      4. Unzip it into:  {ANNOTATED_DIR}

  (B) Programmatic download (this script):
      Grab your API key from  Roboflow -> Settings -> API Keys, then run:

         python roboflow_setup.py --download \\
             --api-key YOUR_KEY --workspace your-workspace \\
             --project cambodian-alpr --version 1

----------------------------------------------------------------------
STEP 8 — Expected directory structure after export (YOLO format)
----------------------------------------------------------------------
  {ANNOTATED_DIR}/
  ├── data.yaml            # class names + train/val/test paths
  ├── train/
  │   ├── images/          # *.jpg
  │   └── labels/          # matching *.txt  (one line per box)
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/

  A label .txt line looks like:
        0 0.5123 0.4780 0.2140 0.0910
        ^  ^      ^      ^      ^
        |  cx     cy     w      h   (all normalised 0-1)
        class id (0 = {CLASS_NAME})

  data.yaml (roughly):
        train: ../train/images
        val:   ../valid/images
        test:  ../test/images
        nc: 1
        names: ['{CLASS_NAME}']

======================================================================
 Once data/annotated/ contains train/valid/test + data.yaml, Week 1
 data prep is DONE. Week 2 = augmentation + YOLOv10 training.
======================================================================
"""


def print_guide() -> None:
    print(textwrap.dedent(GUIDE))


# --------------------------------------------------------------------------- #
# DISCOVERY MODE  (--list)
# --------------------------------------------------------------------------- #

def list_projects(api_key: str, workspace: str | None = None) -> None:
    """Print the active workspace slug, its projects, and their versions.

    Use this when a download fails with 'does not exist or cannot be loaded' —
    it shows the EXACT workspace slug / project id / version number to pass to
    --download.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "The 'roboflow' package is not installed.\n"
            "Activate your venv and run:  pip install roboflow"
        )

    rf = Roboflow(api_key=api_key)
    # No slug -> the account's active/default workspace.
    ws = rf.workspace(workspace) if workspace else rf.workspace()

    print("=" * 60)
    print(f"Workspace slug : {ws.url}")
    print("=" * 60)

    project_ids = ws.projects()  # list of 'workspace/project' identifiers
    if not project_ids:
        print("No projects found in this workspace.")
        print("Create one in the Roboflow UI, then re-run --list.")
        return

    for pid in project_ids:
        short = pid.split("/")[-1]  # the id you pass to --project
        try:
            proj = ws.project(short)
            versions = proj.versions()
            vnums = [v.version.split("/")[-1] for v in versions] or ["(none yet)"]
        except Exception as exc:  # noqa: BLE001 - just for display
            vnums = [f"(could not read versions: {exc})"]
        print(f"\n  project id : {short}")
        print(f"  versions   : {', '.join(vnums)}")

    print("\nCopy the exact 'project id' and a version number into:")
    print("  python roboflow_setup.py --download --api-key KEY \\")
    print(f"      --workspace {ws.url} --project <project id> --version <n>")
    print("\nIf 'versions' is empty, go to Roboflow -> Generate to create one first.")


# --------------------------------------------------------------------------- #
# DOWNLOAD MODE
# --------------------------------------------------------------------------- #

def download_dataset(api_key: str, workspace: str, project: str,
                     version: int, fmt: str = "yolov8") -> None:
    """Pull the annotated dataset from Roboflow into data/annotated/."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "The 'roboflow' package is not installed.\n"
            "Activate your venv and run:  pip install roboflow\n"
            "(setup_week1.py installs it for you.)"
        )

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Roboflow workspace '{workspace}' ...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    print(f"Downloading '{project}' v{version} in '{fmt}' format ...")
    # location= puts the export directly into data/annotated/
    dataset = ver.download(fmt, location=str(ANNOTATED_DIR))

    print("\nDone.")
    print(f"Dataset location : {dataset.location}")
    print(f"Check for        : {ANNOTATED_DIR / 'data.yaml'}")
    print("Next: point YOLOv10 training at that data.yaml in Week 2.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roboflow guide + YOLO-format dataset downloader.")
    parser.add_argument("--download", action="store_true",
                        help="Download an annotated version instead of printing the guide.")
    parser.add_argument("--list", dest="list_", action="store_true",
                        help="List your workspace slug, project ids and versions, then exit.")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    parser.add_argument("--workspace", help="Roboflow workspace slug.")
    parser.add_argument("--project", default="cambodian-alpr",
                        help="Roboflow project id. Default: cambodian-alpr.")
    parser.add_argument("--version", type=int, default=1,
                        help="Dataset version number to download. Default: 1.")
    parser.add_argument("--format", default="yolov8",
                        help="Export format. Default: yolov8 (YOLOv10-compatible labels).")
    args = parser.parse_args()

    if args.list_:
        if not args.api_key:
            raise SystemExit("--list needs --api-key (or ROBOFLOW_API_KEY env var).")
        list_projects(args.api_key, args.workspace)
        return

    if not args.download:
        print_guide()
        return

    missing = [n for n, v in
               (("--api-key", args.api_key), ("--workspace", args.workspace))
               if not v]
    if missing:
        raise SystemExit(
            f"Download mode needs: {', '.join(missing)}.\n"
            "Run without --download to see the full guide, or provide the flags."
        )

    download_dataset(args.api_key, args.workspace, args.project,
                     args.version, args.format)


if __name__ == "__main__":
    main()
