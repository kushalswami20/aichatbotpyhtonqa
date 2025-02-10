# 📜 AI Chatbot - Python Q&A  

## 🚀 Introduction  
This project is an **AI-powered chatbot** that answers **technical questions** related to Python. It uses **T5-based NLP models** to generate responses based on a dataset containing **questions, answers, and tags**.  

---

## 📌 Features  
✔️ **Question-Answering**: Provides answers to Python-related queries.  
✔️ **Fine-Tuned T5 Model**: Uses **T5 (Text-to-Text Transfer Transformer)** for generating responses.  
✔️ **Dataset Merging**: Combines **questions and answers** from multiple sources.  
✔️ **Training & Checkpoints**: Supports training with **gradient accumulation** and saves model checkpoints.  
✔️ **Deployment with Gradio**: Provides a simple **web interface** for users to interact with the chatbot.  

---

## 🛠️ Tech Stack  
- **Python** 🐍  
- **PyTorch** 🔥  
- **Hugging Face Transformers** 🤗  
- **Gradio** 🎨 (for UI)  
- **Google Colab** 💻 (for training)  

---

## 📂 Dataset  
The dataset consists of **questions, answers, and tags** related to Python:  
- `questions.csv` → Contains `Id`, `Title`, `Body`, `Score`  
- `answers.csv` → Contains `ParentId`, `Body`, `Score`  

These datasets are **merged** using `pandas` to create a structured Q&A format.

```python
import pandas as pd

qa_pairs = pd.merge(
    questions[["Id", "Title", "Body", "Score"]],
    answers[["ParentId", "Body", "Score"]],
    left_on="Id",
    right_on="ParentId",
    suffixes=("_question", "_answer"),
)
```

---

## 🏋️‍♂️ Model Training  
The chatbot is trained using **T5-base** with PyTorch.  
### **Training Script (train.py)**  
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load Model & Tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
Training is optimized using **gradient accumulation** to avoid memory overflow.

---


## 🚀 How to Use  
### **1️⃣ Train the Model (Optional)**
```bash
python train.py
```
### **2️⃣ Run the Chatbot**
```bash
python app.py
```
Or deploy in **Colab** using:
```python
!pip install gradio
!python app.py
```

---

## 📌 Future Improvements  
✅ Improve **response accuracy** with fine-tuning  
✅ Add **real-time search** from Python documentation  
✅ Enhance UI with **Streamlit**  

---

