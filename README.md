# NeuroViT: Vision Transformer-based brain tumour diagnosis and medical Q&A using RAG-enhanced LLMs.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Key Tools and Technologies](#key-tools-and-technologies)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Overview

This repository contains an implementation of a Vision Transformer (ViT) model from scratch to classify brain tumors using MRI scans. The project is inspired by the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929), which introduced the Vision Transformer architecture for image classification tasks. The model is trained to classify MRI brain tumor images into four categories: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor. The dataset used for training and evaluation is the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

The model is implemented into the web application, where users can upload images of tumors to have them classified. Once a tumor is detected, the system uses Grad-CAM, from the paper [“Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”](https://arxiv.org/pdf/1610.02391) to generate a heatmap highlighting the tumour region by leveraging the model’s encoder layers. Users can then chat with an AI assistant, powered by Gemma 2B IT, served via vLLM, and enhanced using RAG from Langchain. The information is retrieved from a Vector DB using FAISS, using real scientific literature from PubMed, enabling it to provide accurate and informative medical responses.

Demo: https://youtu.be/dnygzSd4YaE (Originally hosted at http://99.79.114.128/, now permanently offline)

## How It Works

1. **MRI Upload & Tumour Detection**: Users upload brain MRI scans via the web interface. A Vision Transformer model classifies the presence and type of tumour and generates a Grad-CAM heatmap to visually highlight tumor regions in the scan. 
2. **Medical Report Generation**: Once a tumour is detected, the system generates a concise medical report summarizing the findings. This report is supported by relevant scientific literature retrieved through an RAG pipeline, ensuring the information is evidence-based and up-to-date.
3. **AI Doctor Chatbot**: Users can interact with an AI-powered medical chatbot to ask detailed questions about their diagnosis, treatment options, prognosis, and more. The chatbot leverages a large language model (LLM) integrated with the RAG system and a vector database of scientific papers on brain tumours, enabling it to provide accurate, context-aware, and medically informed responses.
4. **Data Handling and Storage**: Uploaded images and chat interactions are processed transiently for inference and are not stored persistently, ensuring user privacy.

Pipeline:
```
[User Upload] -> [ViT Tumour Classifier] -> [Grad-CAM Heatmap] 
      ↓                                            ↓
[LLM Chatbot (RAG)]                        [Report Generator]
```

## Key Tools and Technologies

**Model**

* **PyTorch** – Vision Transformer (ViT-Base, "An Image is Worth 16x16 Words") implemented from scratch with pretrained ImageNet1k weights and fine-tuned for brain tumor classification.
* **Scikit-learn, NumPy, Pandas** – Preprocessing, metrics, and data handling.
* **Matplotlib/Seaborn** – Training and result visualization.
* **ONNX** – Model export for deployment.
* Training used two phases (initial + fine-tuning) with adjusted hyperparameters, data augmentation, and early stopping to prevent overfitting.

**Application**

* **FastAPI** – Backend for MRI upload and results.
* **OpenCV + Grad-CAM** – Heatmap visualizations on scans.
* **LangChain + HuggingFace + FAISS** – RAG pipeline for retrieving PubMed research.
* **vLLM + Gemma 2B IT** – Local LLM serving for the medical chatbot.

**Deployment**

* **Docker + Docker Compose** – Containerized services.
* **AWS EC2 (GPU)** – Cloud hosting with accelerated inference.
* **Nginx + Gunicorn** – Reverse proxy and server.
* Images stored in **ECR**, GPU drivers/toolkits installed, and orchestrated with Docker Compose for scalable deployment.

## Results

Accuracy and Loss history:
![Accuracy & Loss](vit/eda/images/training_graph.png)

F1 Score:  
<img src="vit/eda/images/f1_score.png" alt="F1 Score" width="50%">

Final Model Performance:
```
Train Accuracy: 99.37%
Validation Accuracy: 98.16%
Test Accuracy: 98.93%
```

## Installation and Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/asherk7/NeuroViT.git
   cd NeuroViT
   ```
2. **Train the Model:**
   - Install dependencies, then create the model:
   ```bash
   python vit/create_model.py 
   python utils/convert_model.py
   ```
   - This will train the Vision Transformer model on the brain tumour MRI dataset.
3. **Run the Docker Containers:**
   - Ensure Docker and Docker Compose are installed.
   - Build and run the containers:
   ```bash
   cd docker
   docker-compose up --build
   ```
   - Access the application at `http://127.0.0.1:8000`.
   - Use the web interface to upload MRI scans for tumour detection.
   - Interact with the AI chatbot to ask questions about your diagnosis and treatment options.

### Disclaimer  
The model can be trained on CPU or GPU (recommended); however, the vLLM model can only be deployed using a GPU

### Future Improvements  
- Combine AskLLM and WrapperLLM into one class, refactor the code to use this class in the RAG and regular call
- Add guardrails for images and messages sent that aren't medical-related
- Finding the most probable tumour locations using the average heatmap
  - Hook the last encoder layer and grab the CLS token (1, 197, 768)
  - Use UMAP and reduce dimensionality, use K-Means to cluster images (get clusters of different brain scan angles)
  - For each cluster, calculate a heatmap for each image using gradient values and GRAD-CAM equations
  - Normalize and average the heatmaps, get the most probable tumour location for each view
- Fine-tune the LLM on a medical dataset using QLoRA & HF PEFT, then quantize
- Add the Tumour model to HuggingFace for public use

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)
- [Kaggle Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [PubMed API Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Setup an AI / ML Server From Scratch in AWS With NVIDIA GPU, Docker](https://www.youtube.com/watch?v=N_KFYqvEZvU)
