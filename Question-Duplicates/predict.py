import sys
import nltk
import numpy as np
import pickle
from collections import defaultdict
from trax.supervised import training
from utils import data_generator, siamese


def load_vocab(path="vocab.pkl"):
    with open(path, "rb") as f:
        vocab_regular = pickle.load(f)

    # Restore as defaultdict with OOV handling
    vocab = defaultdict(lambda: 0)
    vocab.update(vocab_regular)
    return vocab


def load_model(model_path="model_3/model.pkl.gz"):
    model = siamese()
    model.init_from_file(model_path, weights_only=True)
    return model

def predict(q1, q2, model, vocab, data_generator, threshold=0.7, verbose=False):
    q1_tokens = nltk.word_tokenize(q1)
    q2_tokens = nltk.word_tokenize(q2)

    Q1 = [vocab[word] for word in q1_tokens]
    Q2 = [vocab[word] for word in q2_tokens]

    Q1, Q2 = next(data_generator([Q1], [Q2], batch_size=1, pad=vocab["<PAD>"], shuffle=False))
    v1, v2 = model([Q1, Q2])
    d = np.dot(v1, v2.T)
    res = d > threshold

    if verbose:
        print(f"Q1: {q1_tokens} → {Q1.tolist()}")
        print(f"Q2: {q2_tokens} → {Q2.tolist()}")
        print(f"Cosine similarity: {d[0][0]:.4f}")
        print("Result:", "✅ Duplicates" if res else "❌ Not duplicates")

    return res

# ----------------------------
def main():
    nltk.download("punkt") 

    print("Enter the first question:")
    q1 = input("> ")

    print("Enter the second question:")
    q2 = input("> ")

    vocab = load_vocab("vocab.pkl")
    model = load_model()
    
    result = predict(q1, q2, model, vocab, data_generator, verbose=True)

    if result:
        print("\n✅ Questions are duplicates.")
    else:
        print("\n❌ Questions are NOT duplicates.")

if __name__ == "__main__":
    main()

