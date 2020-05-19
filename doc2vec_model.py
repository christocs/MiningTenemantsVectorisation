import os, time, csv, re, random
import gensim
import collections

def read_templates(fname, tokens_only=False):
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result = []
        ID_cache = {}
        for i, row in enumerate(reader):
            row_text = re.sub("\*\*.*\*\*('s)?", "", row['Text'])
            tokens = gensim.utils.simple_preprocess(row_text)
            # tag = [float(row['Identifier']) + float(row['Version No']) / 100]
            # tag = [int(row['Identifier']), int(row['Version No']) ]
            # Apparently the tag must be a list with a single, unique integer
            tag = [i]
            result.append(gensim.models.doc2vec.TaggedDocument(tokens, tag))
            ID_cache[i] = row['Identifier'] + '.' + row['Version No']
        return result, ID_cache

def train_model(templates):
    print("Training the model")
    start_time = time.time()
    model = gensim.models.doc2vec.Doc2Vec(documents=templates, vector_size=50, min_count=2, epochs=1000, dm=1)
    assert len(model.docvecs) == len(templates), "Uh oh! The number of trained vectors doesn't equal the number of conditions templates."
    print("Model trained in {0:.2f} seconds\n".format(time.time() - start_time))
    return model

def vector_match(model, templates):
    print("Matching the documents")
    start_time = time.time()

    count_correct = 0
    ranks = []
    second_ranks = []
    first_ranks = []
    for tID in range(len(templates)):
        vector = model.infer_vector(templates[tID].words)
        sims = model.docvecs.most_similar([vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(tID)
        ranks.append(rank)
        if int(float(ID_version[sims[0][0]])) == int(float(ID_version[tID])):
            count_correct += 1 

        first_ranks.append(sims[0])
        second_ranks.append(sims[1])
    
    # Exact matches
    counter = collections.Counter(ranks)
    print(counter)
    # Correct Identifier matches (with different Version numbers)
    print("{} correct out of {} templates... {}%".format(count_correct, len(templates), count_correct / len(templates)))
    print("{0} documents compared in {1:.2f} seconds\n".format(len(ranks), time.time() - start_time))
    return first_ranks, second_ranks


if __name__ == '__main__':
    start_time = time.time()
    templates, ID_version = list(read_templates(os.getcwd() + "/tenement_templates_dupes_removed.csv"))
    print("Imported {0} conditions templates in {1:.2f} seconds\n".format(len(templates), time.time() - start_time))

    #  print(templates[0:10]) # sanity check for correct import
    model = train_model(templates)

    # Assess the model with the training data (sanity check)
    first_ranks, second_ranks = vector_match(model, templates)

    # Similar matches may just be a different Version of the same Identifier
    # Check second best matches that rank highly, but don't have the same Identifier
    for tID in range(len(templates[:10])):
        second = second_ranks[tID]
        if templates[tID].tags is not second[0] and second[1] > 0.9 and int(float(ID_version[tID])) is not int(float(ID_version[second[0]])):
            print('Train Document ({}), ID-{}: «{}»'.format(tID, ID_version[tID], ' '.join(templates[tID].words)))
            print('Similar Document {}, ID-{}: «{}»\n'.format(second[0], ID_version[second[0]], ' '.join(templates[second[0]].words)))

