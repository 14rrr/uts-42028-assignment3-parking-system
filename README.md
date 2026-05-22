# Parkivo – Car Parking Availability Detection

## Overview

Parkivo is a parking availability demo for UTS 42028 Assignment 3. The system
checks which parking spaces are occupied or available from a fixed-camera image
of a monitored parking zone.

The demo uses official PKLot/Voxel51 parking-space polygons as ROIs. Each ROI is
cropped from the full image and passed into a CNN classifier. The Streamlit GUI
then shows the parking status and recommends the best available spaces for a
driver.

## What The App Does

- Runs CNN detection on a parking-zone frame
- Shows occupied and available spaces
- Recommends up to 3 parking spaces
- Shows a simple driver-facing display
- Shows the ROI map and short methodology notes

## Final GUI Tabs

- `Parking Detection`: choose a CNN model, use the default image or upload a
  full-frame parking image, and run detection.
- `Driver Display`: a simple view for drivers after entering the car park,
  showing the best space and other good options.
- `ROI & Methodology`: shows the official ROI map and a short explanation of how
  the system works.

## Models Used

The project compares three CNN models:

- LeNet-5 CNN
- AlexNet CNN
- ResNet-18 CNN

The trained checkpoints are stored in `outputs/checkpoints/`.

## Dataset And ROI

The project uses the PKLot parking dataset. For the GUI demo, the system
monitors one selected parking zone rather than every area in a parking site.

The active ROI map is `assets/custom_slot_map.json`. It contains 68 official
PKLot parking-space polygons for the monitored zone. The default demo image is
`assets/demo_frame_clean.jpg`.

This project is doing image classification on cropped parking-space ROIs. It is
not full-frame object detection.

## How To Run

Open PowerShell and run:

```powershell
cd C:\Users\pc\Documents\GitHub\uts-42028-assignment3-parking-system
C:\Users\pc\miniconda3\envs\ai\python.exe -m streamlit run app_streamlit.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Important Files

- `app_streamlit.py` - Streamlit GUI for the final demo
- `src/` - dataset, model, training, and evaluation code
- `assets/custom_slot_map.json` - official PKLot ROI map used by the GUI
- `assets/demo_frame_clean.jpg` - default clean demo frame
- `outputs/checkpoints/` - trained CNN checkpoints
- `outputs/reports/` - saved Part C result summaries
- `tools/rebuild_official_pklot_roi_map.py` - script used to rebuild the
  official ROI map if needed

## Notes

- The processed dataset and virtual environment are not included in the repo.
- Git LFS is used for the large AlexNet checkpoint.
- The GUI is for assignment and demonstration purposes, not a production parking
  system.
