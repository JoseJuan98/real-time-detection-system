# Real-Time Object Detection and Localization

Real-time object detection and localization using YOLOv11, OpenCV and Faster R-CNN.

## Setup

For the setup of the project follow the instructions in the [SETUP.md](docs/SETUP.md) file.

## Project Structure

```bash
│
├── artifacts                 # folders excluded from the repo, what you store here it won't be store in the repo
│     ├── data
│     └── models
│
├── src                      # source code folder for common code and for CRISP-DM steps
│     ├── common
│     ├── faster_rcnn_experiment  # Faster R-CNN experiment      
│     └── yolo_v11_experiment     # YOLOv11 experiment
│
├── notebooks                 # jupyter notebooks folder
│
├── docs                      # documentation folder 
│     └── Project_Report.md   # final project report
│
├── dev-requirements.txt     # development dependencies
├── environment.yaml         # conda formatted dependencies, used by 'make init' to create the virtualenv
├── README.md                
└── requirements.txt         # core dependencies of the library in pip format
```
