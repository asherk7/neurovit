# NeuroViT-AI: Vision Transformer-based brain tumor diagnosis and medical Q&A using RAG-enhanced LLMs.

A full-stack AI-powered application that enables users to upload brain MRI scans, detect tumors using a Vision Transformer model, and interact with a medical chatbot enhanced by a retrieval-augmented generation (RAG) system. This project combines computer vision, large language models, and modern deployment tools to provide educational insight into brain tumors.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Folder Structure](#folder-structure)
- [Key Tools and Technologies](#key-tools-and-technologies)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Overview

This repository contains an implementation of a Vision Transformer (ViT) model from scratch to classify brain tumors using MRI scans. The project is inspired by the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929), which introduced the Vision Transformer architecture for image classification tasks. The model is trained to classify MRI brain tumor images into four categories: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor. The dataset used for training and evaluation is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

The model is implemented into the web application, where users can upload images of tumors to have them classified. Once a tumor is detected, users can chat with a built-in AI medical assistant to learn more about their diagnosis. The chatbot is powered by a fine-tuned language model served via vLLM, enhanced using RAG (Retrieval-Augmented Generation) and real scientific literature from PubMed, enabling it to provide accurate and informative medical responses.

Demo: (website link or youtube video)

## How It Works

1. **MRI Upload & Tumor Detection**: Users can upload their brain MRI scans through a web interface. The Vision Transformer model processes the image to classify the presence of tumors and draw a bounding box around it.
2. **Medical Report**: After the tumor is detected, the model generates a medical report summarizing the findings and citing relevant scientific literatures, using the Retrieval-Augmented Generation (RAG) approach.
3. **AI Doctor Chatbot**: Users can interact with an AI-powered medical chatbot to ask questions about their diagnosis, treatment options, and more. The chatbot uses a large language model (LLM) finetuned on medical data (MedQA) to provide accurate and helpful responses.
4. **Data Storage**: All uploaded images and generated reports are stored in a secure database, ensuring user privacy and data integrity.

Pipeline:
```
[User Upload] -> [ViT Tumor Classifier] -> [Bounding Box + Classification] 
     ↓                                            ↓
[Report Generator]                         [LLM Chatbot (RAG)]
```

## Folder Structure

```
├── vit/                     # Vision Transformer implementation
│   ├── transformer             # Transformer architecture
│   ├── model                   # Trained weights
│   ├── pipeline                # Training/evaluation pipelines
│   ├── eda                     # Exploratory data analysis
│   └── *.py                    # Utilities and scripts
├── api/                     # FastAPI backend and endpoints
│   ├── core                    # Core logic and service modules
│   ├── routes                  # API route handlers
│   ├── templates               # App UI templates
│   ├── utils                   # Helper functions
│   └── main.py                 # App and server entry point
├── data/                    # MRI dataset
├── llm/                     # vLLM deployment & LangChain integration
├── rag/                     # Vector database & RAG setup
├── scraper/                 # PubMed scraping scripts
├── db/                      # PostgreSQL schema and queries
├── docker/                  # Docker and deployment setup
├── .env                     # Environment variables
├── .gitignore               
├── LICENSE                  
└── README.md
```

## Key Tools and Technologies

### Model

**Technologies**
- **PyTorch**: For implementing and training the Vision Transformer (ViT) model from scratch.
- **Torchvision**: Used for image augmentations and loading the MRI dataset.
- **Scikit-learn**: For preprocessing and evaluation metrics.
- **Matplotlib**: To visualize training performance.
- **Pandas**: For dataset manipulation.

**Training Summary**:

The Vision Transformer (ViT) model is implemented from scratch, following the ViT-Base architecture introduced in ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929). The model was trained in two phases: an initial training phase using pretrained ImageNet1k ViT weights from PyTorch (ViT_B_16_Weights.IMAGENET1K_V1), followed by fine-tuning to optimize performance and accuracy. The implementation closely follows the original architecture, including its optimizer, loss function, and learning rate scheduler, with slight adjustments tailored to the training environment and the smaller dataset size. All modified hyperparameters are documented in `vit/create_model.py`.

- **Initial Training**: Learning rate = `0.0001`, batch size = `32`, epochs = `25`, beta = `(0.9, 0.999)`, weight decay = `0.01`, label smoothing = `0.1`.
- **Fine-tuning**: Learning rate = `0.00001`, epochs = `15`, attention dropout = `0.1`, weight decay = `0.001`.

Data augmentation includes random rotations, flips, shifts, and normalizing. Early stopping is used to avoid overfitting, and the best model is saved for deployment.

### Application

**Technologies**:

- **FastAPI** – Backend framework powering the user interface and API endpoints.
- **OpenCV** – For processing and drawing bounding boxes on MRI scans.
- **LangChain** – Handles the pipeline for RAG (Retrieval-Augmented Generation) in the chatbot.
- **vLLM** – Efficient serving of large language models (LLMs).
- **Hugging Face Transformers, PEFT, QLoRA** – Used to fine-tune and quantize the LLM on the **MedQA** dataset.
- **FAISS** – Enables fast similarity search on embedded scientific documents.
- **Vector Database** – Stores and retrieves scientific literature embeddings for RAG.
- **BeautifulSoup** – Gathers up-to-date biomedical papers from PubMed to power the chatbot.
- **PostgreSQL** – Stores uploaded MRIs and generated diagnostic reports.

**Workflow**:

The application provides an end-to-end pipeline for brain tumor diagnosis and medical assistance. Users upload MRI scans through a FastAPI-powered interface, where the images are processed with OpenCV and analyzed by a custom Vision Transformer (ViT) model trained in PyTorch and fine-tuned on the Brain Tumor MRI dataset. Once a tumor is detected and classified, the system draws bounding boxes on the image and generates a diagnostic report. Users can then interact with an AI medical chatbot, powered by a large language model fine-tuned using Hugging Face Transformers, PEFT, and QLoRA on the MedQA dataset, and served via vLLM for efficient inference. To provide evidence-based responses, the chatbot uses LangChain to perform retrieval-augmented generation (RAG), querying a FAISS-indexed vector database populated with embeddings of PubMed literature scraped using BeautifulSoup. All user data, including uploaded scans and generated reports, is stored securely in PostgreSQL. The entire application is containerized with Docker and deployed on AWS EC2, with Nginx and Gunicorn handling production traffic and Redis used for caching and performance optimization.

### Deployment

**Technologies**:

- **Docker** – Containerizes all components for reproducible deployment.
- **Docker Compose** – Manages multi-service containers (API, DB, LLM server, frontend, etc).
- **AWS EC2** – Hosts the deployed application in the cloud.
- **Nginx** – Acts as a reverse proxy to serve the app.
- **Gunicorn** – Runs the FastAPI app as a WSGI server.
- **Redis** – Used for caching inference results and managing user sessions.

**Deployment Stack**:
The entire stack is containerized using Docker and deployed on AWS EC2. Nginx routes requests, Gunicorn runs the FastAPI backend, and vLLM serves the LLM for chatbot responses. Uploaded images and reports are stored in PostgreSQL. Redis can optionally be added to improve response caching and session performance.

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
     uvicorn api.main:app --reload
     ```
   - Access the application at `http://127.0.0.1:8000`.

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [vLLM Documentation](https://vllm.readthedocs.io/en/latest/)
- [MedQA Dataset](https://paperswithcode.com/dataset/medqa-usmle)
- [PubMed API Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Docker Documentation](https://docs.docker.com/)