# Siamese Network with Hard Negative Mining

This project demonstrates a simple Siamese Network built using [Trax](https://github.com/google/trax). It implements a triplet loss mechanism with **hard negative mining**, combining the **mean of non-duplicates** and the **closest negative sample** as part of the loss function. The model is trained to differentiate between similar and dissimilar pairs of input sequences.

---

## 🛠 Requirements to run project successfully without any errors

- **Python 3.10 ONLY**
- Trax library
- Virtual environment (recommended)

---

## 🚀 Setup Instructions

Follow these steps to run the project in your terminal (Windows):

### 1. 📦 Clone the repository

```
git clone https://github.com/Aliha7ish/Question-Duplicates.git
cd Question-duplicates
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

### 7. 🧠 Use the trained model

Once everything is installed, run the script that uses the trained model (Enter 2 sentences with your own choice and see whether they have the same meaning or not):

```
python predict.py
```
