# MedBrain: Brain Tumor Classification and AI Doctor

A full-stack AI-powered application that enables users to upload brain MRI scans, detect tumors using a Vision Transformer model trained from scratch, and interact with a medical chatbot enhanced by a retrieval-augmented generation (RAG) system. This project combines computer vision, large language models, and modern deployment tools to provide educational insight into brain tumors.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Technologies](#technologies)
- [Model Training](#model-training)
- [Application Deployment](#application-deployment)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Overview (update later when the project is complete)

This repository contains an implementation of a Vision Transformer (ViT) model from scratch to classify brain tumors using MRI scans. The project is inspired by the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929), which introduced the Vision Transformer architecture for image classification tasks. The model is trained to classify MRI brain tumor images into four categories: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor. The dataset used for training and evaluation is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

After detection, the user can interact with a medical chatbot to ask follow-up questions about the diagnosed tumor. The chatbot is powered by an LLM deployed via vLLM, finetuned with domain-specific medical knowledge using the MedQA Dataset, and enhanced with tumor-specific knowledge using RAG (Retrieval-Augmented Generation) and supported by real scientific literature fetched from PubMed.

## How It Works (this is a high-level overview of the application, update later)

1. **MRI Upload & Tumor Detection**: Users can upload their brain MRI scans through a web interface. The Vision Transformer model processes the image to classify the presence of tumors and draw a bounding box around it.
2. **Medical Report**: After the tumor is detected, the model generates a medical report summarizing the findings and citing relevant scientific literatures, using the Retrieval-Augmented Generation (RAG) approach.
3. **AI Doctor Chatbot**: Users can interact with an AI-powered medical chatbot to ask questions about their diagnosis, treatment options, and more. The chatbot uses a large language model (LLM) finetuned on medical data (MedQA) to provide accurate and helpful responses.
4. **Data Storage**: All uploaded images and generated reports are stored in a secure database, ensuring user privacy and data integrity.

## Folder Structure (Update later)

```
├── vit/                     # Custom ViT implementation
│   ├── transformer          # Transformer architecture implementation
│   ├── model                # Vision Transformer model weights
│   ├── pipeline             # Training and evaluation pipelines for the ViT model
│   ├── eda                  # Exploratory Data Analysis scripts/images for the ViT model
│   └── *.py                 # Other scripts related to the ViT model (e.g. data loading, utils, main training script)
├── api/                     # FastAPI for web app, model inference, and chatbot
├── data/                    # Dataset for training and evaluation
├── llm/                     # LLM deployment with vLLM and LangChain integration
├── rag/                     # Scripts for vector DB and LangChain setup
├── scraper/                 # PubMed paper retrieval script
├── db/                      # PostgreSQL schema and logging scripts
├── docker/                  # Docker setup for full app
├── utils/                   # Utility scripts 
├── .env                     # Environment variables for configuration
├── .gitignore               
├── LICENSE                  
└── README.md
```

## Key Tools and Technologies (update with scraper technology)
## combine model training and application/deployement into this section

### Model
- **PyTorch**: Deep learning framework used for model implementation and training.
- **Torchvision**: Library for computer vision tasks, used for data transformations and augmentations.
- **Scikit-learn**: Library for machine learning, used for data preprocessing and evaluation metrics.
- **Matplotlib**: Library for data visualization, used to plot training metrics and model performance.
- **Pandas**: Data manipulation library, used for handling datasets.
### Application
- **FastAPI**: Web framework for building the application interface.
- **vLLM**: Library for deploying large language models efficiently.
- **LangChain**: Framework for building Retrieval-Augmented Generation (RAG's) with LLMs, used to create the AI doctor chatbot.
- **FAISS and vector db go here**: Libraries for efficient similarity search and retrieval of embeddings, used to store and retrieve scientific literature embeddings.
- **Insert scraper technology here**: Tool for scraping scientific literature from PubMed to enhance the chatbot's knowledge base.
- **Docker**: Containerization tool for deploying the application.
- **PostgreSQL**: Database for storing user-uploaded images and generated reports.
- **OpenCV**: Library for image processing, used for handling and manipulating MRI scans.
### Deployment
- **Docker Compose**: Tool for defining and running multi-container Docker applications.
- **AWS EC2**: Cloud service for hosting the application.
- **Nginx**: Web server for serving the application.
- **Gunicorn**: WSGI HTTP server for running the FastAPI application.
- **Redis (maybe)**: In-memory data structure store, used for caching and session management.

## Model Training (talk about paper implementation, adjusting hyperparameters, loading weights and finetuning, optimization, etc.)

The Vision Transformer model is implemented from scratch, inspired by the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929). The model architecture consists of:
- **Patch Embedding**: The input image is divided into fixed-size patches, which are then flattened and linearly projected to create patch embeddings.
- **Transformer Encoder**: The patch embeddings are passed through a series of transformer encoder layers, which apply self-attention mechanisms to capture long-range dependencies in the image.
- **Classification Head**: The output of the transformer encoder is passed through a classification head to produce the final class probabilities.
The model is trained on the Brain Tumor MRI dataset using the following hyperparameters:
- **Learning Rate**: 0.0001
- **Batch Size**: 32 
- **Number of Epochs**: 25
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Scheduler**: cosine annealing with warm restarts
The model was finetuned after with the following hyperparameters:
- **Learning Rate**: 0.00005
- **Number of Epochs**: 15
- **Label Smoothing**: 0.1

The model is trained using the training set, and its performance is evaluated on the validation set. The training process includes data augmentation techniques such as random rotations, flips, and color jittering to improve model generalization.

To maximize performance, the model weights are initialized using a pre-trained ViT model from the PyTorch library, trained on ImageNet1k (go more in-depth here). The model is then fine-tuned on the brain tumor dataset, adjusting hyperparameters such as learning rate and batch size based on validation performance. The training process includes early stopping to prevent overfitting, and the best model weights are saved for later use in the application.

## Application Deployment (should go in depth on all the tools and how they are used)
(list the main features and functionalities (e.g. rag, vector db, FAISS (how embeddings are stored/retrieved) llm chosen, script, storage, etc.))

The application is built using FastAPI, which provides a modern web interface for users to interact with the AI doctor chatbot and upload MRI scans. The main features of the application include:
- **MRI Upload**: Users can upload their brain MRI scans through a user-friendly interface.
- **Tumor Detection**: The Vision Transformer model processes the uploaded MRI scans to detect and classify brain tumors, drawing bounding boxes around detected tumors using OpenCV.
- **Medical Report Generation**: After tumor detection, the model generates a medical report summarizing the findings and citing relevant scientific literature using the RAG approach.
- **AI Doctor Chatbot**: Users can interact with an AI-powered medical chatbot to ask questions about their diagnosis, treatment options, and more. The chatbot uses a large language model (LLM) finetuned on medical data (MedQA) to provide accurate and helpful responses.
- IMPORTANT: mention how the llm from vllm was finetuned (using huggingface PEFT, QLoRA, etc.) and how the model is loaded in the app
- **LangChain Integration**: The application uses LangChain to manage the interaction between the LLM and the retrieval system, allowing the chatbot to fetch relevant information from a vector database.
- **Vector Database**: The application uses a vector database to store and retrieve embeddings of scientific literature, enabling the chatbot to provide accurate and up-to-date information.
- **PubMed Scraper**: The application includes a scraper that fetches relevant scientific literature from PubMed to enhance the chatbot's knowledge base.
- **Retrieval-Augmented Generation (RAG)**: The chatbot is enhanced with RAG capabilities, allowing it to fetch relevant scientific literature from PubMed to provide accurate and up-to-date information.
- **Data Storage**: All uploaded images and generated reports are stored in a secure PostgreSQL database, ensuring user privacy and data integrity.
- **Dockerized Deployment**: The application is containerized using Docker, allowing for easy deployment and scalability.
- **AWS Deployment**: The application is deployed on AWS EC2, with Nginx serving as the web server and Gunicorn running the FastAPI application. Redis is used for caching and session management, improving application performance.
- **consider adding redis, etc. here**

## Results

Accuracy and Loss history:
![Accuracy & Loss](vit/eda/images/training_graph.png)

F1 Score:
![F1 Score](vit/eda/images/f1_score.png)

Model Summary: 
![Model Summary](vit/eda/images/model_summary.png)

Final Model Performance:
```
Train Accuracy: 99.50%
Validation Accuracy: 97.90%
Test Accuracy: 98.17%
```

## Installation and Usage (update when app is made)
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/asherk7/MedBrain.git
   cd MedBrain
   ```
2. **Train the Model**:
   - Run the training script: (Ensure necessary dependencies & dataset are installed)
     ```bash
     cd utils
     python create_model.py
     ```
   - This will train the Vision Transformer model on the brain tumor MRI dataset.
3. **Run the Application**:
   - Start the FastAPI application:
     ```bash
     uvicorn main:app --reload
     ```
   - Access the application at `http://localhost:8000`.

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [vLLM Documentation](https://vllm.readthedocs.io/en/latest/)
- [MedQA Dataset](https://paperswithcode.com/dataset/medqa-usmle)
- [PubMed API Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Docker Documentation](https://docs.docker.com/)