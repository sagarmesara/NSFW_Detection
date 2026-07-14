# 🛡️ NSFW Image Content Filter

An end-to-end Deep Learning pipeline built to classify images as Safe For Work (SFW) or Not Safe For Work (NSFW). This project covers the entire machine learning lifecycle: from raw data ingestion and preprocessing to Convolutional Neural Network (CNN) training, and finally, deployment via an interactive Streamlit web application.

## 🌐 Live Demo

Experience the filter in action: [NSFW Content Detector](https://nsfw-detection-app.streamlit.app/)
* Note: This application processes images in-memory for privacy and performance.

## 🚀 Project Overview

* **Frameworks:** TensorFlow / Keras, OpenCV, Streamlit
* **Architecture:** Custom CNN with Data Augmentation and Global Average Pooling
* **Performance:** ~80-82% Accuracy (configurable via training epochs)
* **Deployment:** Real-time inference through a Streamlit UI handling in-memory byte buffers.

---

## 📂 Repository Structure

The workflow is broken down into modular Jupyter Notebooks and a deployment script.

* `data_arranging.ipynb`: Scans the raw dataset, maps 5 distinct sub-categories into a binary classification structure (SFW vs. NSFW), shuffles the data, and splits it 80/20 into training and validation sets.
* `model_training.ipynb`: Defines the CNN architecture, applies on-the-fly data augmentation, compiles the model with false-negative tracking, and trains the network. Outputs `NSFW_predictor.keras`.
* `prediction.ipynb`: A testing sandbox for running inference on local image paths using OpenCV preprocessing.
* `app.py`: The frontend web application built with Streamlit. Allows users to upload images and receive real-time classification verdicts.

---

## 🗄️ Dataset Structure

To train this model from scratch, you need to provide a raw dataset organized into specific subfolders. 

### 1. Raw Ingestion Format
Create a folder named `nsfw_dataset` in the root directory. Place your images into the following subfolders based on their content:

```text
nsfw_dataset/
├── drawings/   (Mapped to SFW)
├── neutral/    (Mapped to SFW)
├── hentai/     (Mapped to NSFW)
├── porn/       (Mapped to NSFW)
└── sexy/       (Mapped to NSFW)

### 2. Running data_arranging.ipynb will automatically parse the raw folders and generate the standard Keras image_dataset_from_directory structure:

nsfw_project_dataset/
├── train/
│   ├── nsfw/   (Class 0)
│   └── sfw/    (Class 1)
└── val/
    ├── nsfw/
    └── sfw/