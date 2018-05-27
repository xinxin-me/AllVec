# AllVec
Training algorithm for learning word embedding from all samples. Please use the GloVe code to build co-occurrence statistics.

word_vetors_v1 utilize gradient descent methods and word_vectors_v2 utilize Newton methods.

Examples to run the code:

./word_vectors_v2 -word-occu $co-occurrence file$ -read-vocab $vocab_file$ -output ./vectors.bin -w0 350 -size 50 -iter 50 -binary 1 -threads 16 -thro 0.8 -shift -0.5
