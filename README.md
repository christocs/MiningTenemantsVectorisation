# Vectorisation of the conditions

As at the time of writing, you may run `python3 word2vec_model.py` in this dir to generate a model, which gets saved as a file. 

Running `python3 word2vec_model.py <temp-file-name>` will load the model again. 

## Ideas for improvements

In no particular order:

* It's possible that the large number of duplicate conditions distorts the weights of the individual words. In pre-processing, we should consider removing duplicate conditions from when training the word2vec model.
* There may be some value in doc2vec
* the `visualise_model.py` file doesn't seem to generate a graph atm. See https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py 

