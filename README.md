# LT2222 V21 Assignment 3

Your name: Jacob Coles

## Part 1

The function a() takes a file f, and generates a list  mm, where each item in mm is a character from the text (with two additional start <s> and end <e> end characters/tags). 
It returns a tuple containing mm, and a listified set of mm (list of all the possible unique items in mm). 

The function g() is called inside the next function b(). Funciton g() takes two arguments; a character x and list of possible characters p. It then generates a list of zeroes of length of p, for one-hot-encoding of the character x. It generates the one-hot-encode of the single character x. It then returns it. 

Function b() takes two arguments; u which is a list of all characters in the text (in order), and p which is the unique set of characters in the text. (This corresponds to the output of function a()).
It iterates through every vowel character and does two things. Firstly, it appends that vowel to list gt (however, stored as an index with reference to the list called vowels). Secondly, it goes through the contextual characters/tags either side of each vowel, and uses function g() to create one-hot-encodes of these contextual words, where the one-hot-encodes of all the context characters associated with each individual vowel are concatenated, and appended to array gr as a single vector.

In the main block of code we:
    q = a(args.m) #gets a list of characters in the text, and the set of unique characters
    w = b(q[0], q[1]) #takes q and generates list of one-hot-encodes of the context of each vowel, and a corresponding list of each vowel itself
    t = train(w[0], w[1], q[1], args.k, args.r) #train(X, y, vocab, hiddensize, epochs=100)
    using 'input' w[0] (the context of the vowels), and the 'output' w[1] (the vowels themselves), train a model to predict the vowel from the context. We also pass in q[1] which is the set of all unique characters in the text, k, which is the hidden-layer size, and r which is the number of epochs. 


## Part 2
GENERATE MODEL:
To generate the model from the training data, run:
python3 train.py /path/to/training/set model_file
(for example):
python3 train.py /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtrain.lower.txt model_file
Additionally there are two optional arguments:
    --k, an integer setting the number of hidden layers, with default=200
    --r, an integer setting the number of epochs/training iterations, with default=100
    parser.add_argument("--r", dest="r", type=int, default=100)

GENERATE OUTPUT FROM FROM TEST DATA:
To generate the file with replaced (generated) vowels based on the test data, run:
python3 eval.py /path/to/training/set /path/to/test/set ./model_file
(for example):
python3 eval.py /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtrain.lower.txt /home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtest.lower.txt ./output_model generated_vowels.txt

## Part 3
Accuracy results for variations of --k and --m options
Variations of --k (layers) option:
10: 0.1327358860625811
50: 0.1443680033794997
100: 0.12095289822274524
200: 0.1535258441205757
400: 0.1274855918650614
1000: 0.1548685917745391

Variations of --r (epochs) option:
10: 0.14043028273135993
50: 0.17307866389064905
100: 0.1535258441205757
200: 0.1466310612232582
400: 0.1650070909145771
1000: 0.1507196523943152

With each parameter change, it seems to fluctuate up and down in accuracy between around 0.12 to 0.17 (approximately). Looking at the texts they output, generally all the vowels end up getting turned into one specific vowel, which can vary randomly. Somehow the model seems to select a vowel which it deems as most likely, and applies it to all the positions with a vowel. These accuracy percantages seem to show the proportion/frequency of the 'selected' vowel across the vowels in the text. 

## Bonuses

Part C:
Modified the VowelModel class in model.py to include dropout. The new version is model_dropout.py .
The version with dropout can be run by adding a --d flag when training (when you run train.py). 
Following are the results in the same format as Part 3, except when training the model with dropout:

Accuracy results for variations of --k and --m options
Variations of --k (layers) option:
10: 0.14014362873781722
50: 0.11030143930478863
100: 0.15212274825744546
200: 0.12256721281795963
400: 0.10245617211309255
1000: 0.12486044476630156

Variations of --r (epochs) option:
10: 0.11627591201231104
50: 0.11600434507106001
100: 0.12256721281795963
200: 0.12120937811170454
400: 0.13355058688633414
1000: 0.16055641048851876

There seems to be higher stability when including dropout. 

## Other notes
