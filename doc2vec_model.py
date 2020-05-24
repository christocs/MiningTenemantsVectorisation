import os, time, csv, re, random
import gensim
from gensim.models.doc2vec import TaggedDocument
import collections

NEAR_MATCH_THRESHOLD = 0.03 # how close a template match must be to the highest match to be considered significant
EPOCHS = 100

class Template(TaggedDocument):
    def __new__(self, tokens, tags, id, version):
        obj = TaggedDocument.__new__(self, tokens, tags)
        obj.id = id
        obj.version = version
        return obj

def read_conditions(fname, condition_column_name):
    start_time = time.time()
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tokens = gensim.utils.simple_preprocess(row[condition_column_name])
            result.append(tokens)
    print("Imported {0} conditions in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
    return result

def read_templates(fname, template_column_name = "Text"):
    start_time = time.time()
    result = []
    ID_cache = {}
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_text = re.sub("\*\*.*\*\*('s)?", "", row[template_column_name])
            tokens = gensim.utils.simple_preprocess(row_text)
            result.append(Template(tokens, [i], int(row['Identifier']), int(row['Version No'])))
            ID_cache[i] = row['Identifier'] + '.' + row['Version No']
    print("Imported {0} conditions templates in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
    return result

def train_model(templates):
    print("Training the model")
    start_time = time.time()
    model = gensim.models.doc2vec.Doc2Vec(documents=templates, vector_size=50, min_count=2, epochs=EPOCHS, dm=1)
    assert len(model.docvecs) == len(templates), "Uh oh! The number of trained vectors doesn't equal the number of conditions templates."
    print("Model trained in {0:.2f} seconds\n".format(time.time() - start_time))
    return model


def vector_match(model, templates, conditions = []): 
    print("Matching the documents")
    start_time = time.time()

    # conditions is expected to be a list of list of words
    if len(conditions) == 0: # if no conditions are supplied, just match templates against themselves
        conditions = [" ".join(t.words) for t in templates]
    single_matches = {}
    multiple_matches = {}
    for cID, condition in enumerate(conditions):
        condition_vector = model.infer_vector(condition)
        most_similar = model.docvecs.most_similar([condition_vector], topn=5) 
        highest_score = most_similar[0][1]

        # determine how many other templates rank close to this one
        # TODO: this should probably only consider DIFFERENT TEMPLATE IDs, not different versions of the same ID
        num_close_matches = sum(1 for tID, score in most_similar[1:] if abs(highest_score - score) < NEAR_MATCH_THRESHOLD)
        
        if num_close_matches == 0:
            highest = templates[most_similar[0][0]]
            single_matches[cID] = (highest_score, highest.id, highest.version)
        else:
            multiple_matches[cID] = [(sim[1], templates[sim[0]].id, templates[sim[0]].version) for sim in most_similar[0:num_close_matches+1]]
    print("{0} documents compared in {1:.2f} seconds\n".format(len(conditions), time.time() - start_time))
    return single_matches, multiple_matches


if __name__ == '__main__':
    templates = read_templates(os.getcwd() + "/tenement_templates_dupes_removed.csv")
    conditions = read_conditions(os.getcwd() + "/ConditionsNoRegexMatches.csv", "CondText")
    model = train_model(templates)

    # Assess the model with the training data (sanity check)
    matched, uncertain = vector_match(model, templates, conditions)
    print("{0} documents matches exactly, {1} have multiple matches\n".format(len(matched), len(uncertain)))
    
    ranks = []
    for ID, matches in uncertain.items():
        ranks.append(len(matches))
    counter = collections.Counter(ranks)
    print("The number of matches per uncertain document are:")
    print(counter)
