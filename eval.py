import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from model import train
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def split_file_to_chars(file):
    chars = []
    with open(file, "r") as f:
        for line in f:
            chars += [c for c in line]

    chars = ["<s>", "<s>"] + chars + ["<e>", "<e>"]
    return chars, list(set(chars))

def chars_to_file_w_vowel_replacement(chars, new_vowels, file_out):
    all_text = str()
    
    for v in range(len(chars) - 4): 
        new_vowel_index = 0
        if chars[v+2] not in vowels:
            all_text += chars[v+2]
            continue
        all_text += vowels[new_vowels[new_vowel_index]]
        new_vowel_index += 1
    
    f = open(file_out, "w")
    f.write(all_text)
    f.close()


def create_one_hot(char, unique_chars):
    hot_vector = np.zeros(len(unique_chars))
    try:
        hot_vector[unique_chars.index(char)] = 1
    except:
        pass
    return hot_vector

def vowel_w_context(chars, unique_chars):
    vowels_current = []
    vowel_contexts = []
    for v in range(len(chars) - 4): 
        if chars[v+2] not in vowels:
            continue
        
        vowel_current = vowels.index(chars[v+2])
        vowels_current.append(vowel_current)
        vowel_context = np.concatenate([create_one_hot(char, unique_chars) for char in [chars[v], chars[v+1], chars[v+3], chars[v+4]]])
        vowel_contexts.append(vowel_context)

    return np.array(vowel_contexts), np.array(vowels_current)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="k", type=int, default=200)
    parser.add_argument("--r", dest="r", type=int, default=100)
    parser.add_argument("train_file_path", type=str) #file input for training data
    parser.add_argument("test_file_path", type=str) #file input for test data
    parser.add_argument("model_file_path", type=str) #file input for pytorch model
    parser.add_argument("output_file_path", type=str) #file output for generated
    
    args = parser.parse_args()

    train_chars, train_unique_chars = split_file_to_chars(args.train_file_path)
    train_X, train_y = vowel_w_context(train_chars, train_unique_chars)

    test_chars, __test_unique_chars = split_file_to_chars(args.test_file_path)
    test_X, test_y = vowel_w_context(test_chars, train_unique_chars)
    
    model = torch.load(args.model_file_path)

    model.eval()
    
    test_X_t = torch.LongTensor(test_X)
    test_X_t.size()
    
    train_X_t = torch.LongTensor(train_X)
    train_X_t.size()
    
    outputs = model(test_X_t.float().unsqueeze(0))
    
    predictions = pd.Series(outputs.squeeze(0).argmax(dim=1).numpy())
    
    chars_to_file_w_vowel_replacement(test_chars, predictions, args.output_file_path)
    
    accuracy = len(test_y[predictions == test_y]) / len(test_y)
    print(accuracy)
    