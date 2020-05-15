# Vectorisation of the conditions

## Doc2Vec
Currently the doc2vec model can be run using `python3 doc2vec_model.py` in the repo directory. It builds a model based on the conditions templates, and then assesses the accuracy against the same training data. The code structure follows is modelled on the [official tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html).

Since there are many duplicate or very similar conditions the accuracy is not great. To improve this, we may need to manually sort through the templates to group/remove duplicates.

## Word2Vec
As at the time of writing, you may run `python3 word2vec_model.py` in this dir to generate a model, which gets saved as a file. 

Running `python3 word2vec_model.py <temp-file-name>` will load the model again. 

### Ideas for improvements

In no particular order:

* It's possible that the large number of duplicate conditions distorts the weights of the individual words. In pre-processing, we should consider removing duplicate conditions from when training the word2vec model.
* There may be some value in doc2vec
* the `visualise_model.py` file doesn't seem to generate a graph atm. See https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#visualising-the-word-embeddings

