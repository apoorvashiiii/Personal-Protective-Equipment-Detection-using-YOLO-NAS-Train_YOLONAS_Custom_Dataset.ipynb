
# PPE Detection using YOLO-NAS

**Author:** Apoorva Singh

**Registration No.:** 24bac10011

This project implements a computer vision pipeline for detecting Personal Protective Equipment (PPE) on construction sites or industrial floors. The notebook is configured to train a detection model using the SuperGradients framework and is designed to be executed in a Google Colab environment with GPU acceleration.

## Dataset Integration

The project utilizes a custom dataset imported directly into the Colab environment via the Roboflow API. The data is downloaded in the YOLOv5 PyTorch format and split into training, validation, and testing directories.

**Detection Classes:**

* Dust Mask


* Eye Wear


* Glove


* Protective Boots


* Protective Helmet


* Safety Vest


* Shield



To improve model generalization, the training data pipeline applies a robust suite of augmentations, including `DetectionMosaic`, `DetectionRandomAffine` (customized to 10.42 degrees), `DetectionMixup`, `DetectionHSV`, and `DetectionHorizontalFlip`.

## Prerequisites & Installation

The environment relies on the following core libraries:

* `super-gradients==3.1.0`

* `roboflow`

* `imutils`

* `pytube` (upgraded)



## Usage Instructions

1. **Hardware Accelerator:** Before executing any code, verify that the Colab Runtime is set to **GPU**.


2. **Setup Environment:** Run the initial cell to install all required pip packages.


3. **Restart Runtime:** You **must** restart the Colab runtime immediately after package installation to prevent dependency conflicts.


4. **Initialize Trainer:** The script instantiates a SuperGradients `Trainer` saving to the `checkpoints` directory under the experiment name `ppe_yolonas_run`.


5. **Prepare Dataloaders:** The downloaded dataset is mapped using `coco_detection_yolo_format_train` and `coco_detection_yolo_format_val` with a batch size of 16 and 2 workers.



