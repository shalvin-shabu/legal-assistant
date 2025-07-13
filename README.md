# Legal Assistant

This model demonstrates a fine-tuned **T5 transformer model** for simplifying legal English into plain English using the Hugging Face `transformers` and `datasets` libraries and then translated to malayalm using MarianMT (by Helsinki-NLP)

---

## 📌 Project Overview

Legal documents are often difficult to understand due to their formal and complex structure. This project fine-tunes a **T5-small** model to transform complex legal clauses into simpler, more accessible language.

---

## 📁 Files

- `L4_cleaned.ipynb` — Jupyter notebook containing the entire code for data preparation, model fine-tuning, and saving the model.
- `legal_t5_model/` — Directory (created by the notebook) that contains the trained model and tokenizer.
- `legal_t5_model.zip` — Zipped version of the fine-tuned model for easy download and sharing.

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

---

### 2. Install Dependencies
pip install transformers datasets sentencepiece

Model: T5-small

Model Type: Encoder-Decoder (Seq2Seq)

Base: t5-small

Task Format: Prefix each input with simplify: to instruct the model.

---

🧪 Dataset
A small toy dataset is created inside the notebook using the Hugging Face datasets.Dataset.from_dict() method with legal clauses and their simplified versions.

Example:

Input (Legal Text) 
simplify: The lessee shall remit payment within thirty (30) days of receipt.        	

Output (Simplified Text)
The renter must pay within 30 days.

---

### Training Details

Tokenizer: T5Tokenizer

Model: T5ForConditionalGeneration

Epochs: 50

Batch Size: 2

Evaluation Strategy: Epoch-based

---

### Output
Trained model is saved to ./legal_t5_model/

A ZIP archive legal_t5_model.zip is created and can be downloaded directly.

---

### How to Use the Model
## 🧪 Example Usage

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("legal_t5_model")
tokenizer = T5Tokenizer.from_pretrained("legal_t5_model")

# Define a simplification function
def simplify(text):
    inputs = tokenizer("simplify: " + text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
simplified = simplify("The lessee shall remit payment within thirty (30) days of receipt.")
print("Simplified Text:", simplified)


simplify("The lessee shall remit payment within thirty (30) days of receipt.")
```
---

### Notes
This is a proof-of-concept using synthetic toy data.

For production use, consider training on larger datasets like contracts or public case law documents.

The model works best when you prefix input with simplify: as shown above.

---

### License
This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

### Contributions
Pull requests and suggestions are welcome!
