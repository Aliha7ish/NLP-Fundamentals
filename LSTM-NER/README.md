# Named Entity Recognition

This project demonstrates a simple Named Entity Recognition model built using [Trax](https://github.com/google/trax) with the powerful tool LSTM Architicure. It takes a sentence as input and outputs the named entity tags for each word.

---

## 📁 Project Structure

LSTM-NER/
│
├── data/
│ └── large/
│ ├── words.txt
│ └── tags.txt
│ └── eval/
│ └── small/
│ └── train/
├── model_3/
├── ner_predict.py
├── utils_copy_1.py
├── NER LSTM.ipynp
├── ner_dataset
├── requirements.txt
└── README.md

---

## 🛠 Requirements to run project succefuly without any errors

- **Python 3.10 ONLY**
- Trax library
- Virtual environment (recommended)

---

## 🚀 Setup Instructions

Follow these steps to run the project in your terminal (Windows):

### 1. 📦 Clone the repository

```
git clone https://github.com/Aliha7ish/LSTM-NER.git
cd LSTM-NER
```

### 2. 🐍 Create a virtual environment with Python 3.10

Make sure Python 3.10 is installed on your system.

```
# Replace the path with the actual Python 3.10 path if needed
"C:\Path\To\Python310\python.exe" -m venv traxenv
```

### 3. ✅ Activate the environment

```
traxenv\Scripts\activate
```

### 4. 🔍 Confirm Python version

inside cmd you can paste this commands

```
python
import sys
print(sys.version)
```

if the output shows Python 3.10.x and does not say "Anaconda" then we are on the correct path

### 5. 📉 Downgrade conflicting packages

```
pip install protobuf==3.20.3 --force-reinstall
pip install numpy==1.26.4 --force-reinstall
```

### 6. ⬆️ Upgrade pip & Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 7- Use the model on your sentence

Once everything is installed, use the ner_predict.py script from the command line to test the model:

```
python ner_predict.py --sentence "I live in Egypt"
```
