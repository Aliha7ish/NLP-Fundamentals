# Import packages
import numpy as np
import os
import trax
from trax import layers as tl
from utils_copy_1 import get_params, get_vocab
import argparse

# Load preprocessed data
vocab, tag_map = get_vocab('data/large/words.txt', 'data/large/tags.txt')
vocab_size = len(vocab)

# define lstm model
def LSTM(vocab_size = vocab_size, d_model = 128, tags = tag_map):
    model = tl.Serial(
        tl.Embedding(vocab_size, d_model),
        tl.LSTM(n_units = d_model),
        tl.Dense(len(tags)),
        tl.LogSoftmax()
    )
    return model

# define load_trained_model function
def load_trained_model(model_path, vocab_size, tag_map, d_model=128, seq_len=10):
    model = LSTM(vocab_size=vocab_size, d_model=d_model, tags=tag_map)
    model.init(trax.shapes.ShapeDtype((1, seq_len), dtype=np.int32))
    model.init_from_file(model_path, weights_only=True)
    return model

# define predict function
def predict(sent, model, vocab, tags):
    # convert tokens to vocab indices
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sent.split(' ')]
    batch = np.ones((1, len(s)))
    batch[0][:] = s
    sentence = np.array(batch).astype(int)

    # get model predictions
    output = model(sentence)
    outputs = np.argmax(output, axis=2)

    # map predictions to tag labels
    labels = list(tags.keys())
    pred = [labels[outputs[0][i]] for i in range(len(outputs[0]))]
    return pred

def main():
    parser = argparse.ArgumentParser(description="NER CLI using Trax LSTM")
    parser.add_argument('--model_path', type=str, default='./model_3/model.pkl.gz', help='Path to saved model')
    parser.add_argument('--sentence', type=str, help='Sentence to tag')
    args = parser.parse_args()

    if not args.sentence:
        print("❌ Please provide a sentence using --sentence.")
        return

    sentence = args.sentence.strip()
    words = sentence.split()

    # Load model
    model = load_trained_model(args.model_path, vocab_size=vocab_size, tag_map=tag_map, seq_len=len(words))

    # Predict
    tags = predict(sentence, model, vocab, tag_map)

    # Print results
    print("\n Predicted NER Tags:")
    for word, tag in zip(words, tags):
        print(f"{word:10} → {tag}")

if __name__ == '__main__':
    main()