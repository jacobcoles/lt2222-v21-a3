import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from model import train
from model_dropout import train_dropout
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])
 
def a(f):
    mm = []
    with open(f, "r") as q:
        for l in q:
            mm += [c for c in l]

    mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"]
    return mm, list(set(mm))

def g(x, p):
    z = np.zeros(len(p))
    z[p.index(x)] = 1
    return z


def b(u, p):
    gt = []
    gr = []
    for v in range(len(u) - 4): 
        if u[v+2] not in vowels:
            continue
        
        h2 = vowels.index(u[v+2])
        gt.append(h2)
        r = np.concatenate([g(x, p) for x in [u[v], u[v+1], u[v+3], u[v+4]]])
        gr.append(r)

    return np.array(gr), np.array(gt)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="k", type=int, default=200)
    parser.add_argument("--r", dest="r", type=int, default=100)
    parser.add_argument("m", type=str) #file name input for training data
    parser.add_argument("h", type=str) #file name output for output model
    parser.add_argument('--d', dest='dropout_flag', action='store_true')
    
    args = parser.parse_args()

    q = a(args.m)
    w = b(q[0], q[1])
    
    if args.dropout_flag:
        t = train_dropout(w[0], w[1], q[1], args.k, args.r) 
    else:
        t = train(w[0], w[1], q[1], args.k, args.r) 

    torch.save(t, args.h)
