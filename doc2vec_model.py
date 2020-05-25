import os, time, csv, re, random, pickle, collections
import gensim
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

NEAR_MATCH_THRESHOLD = 0.03 # how close a template match must be to the highest match to be considered significant
EPOCHS = 200

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
            result.append(row[condition_column_name])
    print("Imported {0} conditions in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
    return result

def tokenise(collection):
    start_time = time.time()
    result = [gensim.utils.simple_preprocess(row) for row in collection]
    print("Preprocessed {0} conditions in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
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
        most_similar = model.docvecs.most_similar([condition_vector], topn=10) 
        highest_score = most_similar[0][1]

        # determine how many other templates rank close to this one
        # only consider DIFFERENT TEMPLATE IDs, not different versions of the same ID
        similar_template_ids = [ templates[most_similar[0][0]].id ]
        num_close_matches = 0
        for tID, score in most_similar:
            current = templates[tID]
            if abs(highest_score - score) < NEAR_MATCH_THRESHOLD and current.id not in similar_template_ids:
                num_close_matches = num_close_matches + 1
        
        if num_close_matches == 0:
            highest = templates[most_similar[0][0]]
            single_matches[cID] = (highest_score, highest.id, highest.version)
        else:
            multiple_matches[cID] = [(sim[1], templates[sim[0]].id, templates[sim[0]].version) for sim in most_similar[0:num_close_matches+1]]
    print("{0} documents compared in {1:.2f} seconds\n".format(len(conditions), time.time() - start_time))
    return single_matches, multiple_matches


if __name__ == '__main__':
    templates = read_templates(os.getcwd() + "/tenement_templates_dupes_removed.csv")
    orig_conditions = read_conditions(os.getcwd() + "/ConditionsNoRegexMatches.csv", "CondText")
    conditions = tokenise(orig_conditions)
            
    model = train_model(templates)

    # Assess the model with the training data (sanity check)
    matched, uncertain = vector_match(model, templates, conditions)
    print("{0} documents have been paired to a template ID exactly, {1} have multiple matches\n".format(len(matched), len(uncertain)))
    
    ranks = []
    for ID, matches in uncertain.items():
        ranks.append(len(matches))
    counter = collections.Counter(ranks)
    print("The number of matches per uncertain document are:")
    print(counter)

    # TODO: rather than just report the mean, this should output a distribution, preferrable a plot
    #       if skewed, should report the median and interquartile range
    matched_scores = [m[0] for k,m in matched.items()]
    matched_mult_high_scores = [u[0][0] for k,u in uncertain.items()]
    avg_matched_score = sum(matched_scores) / len(matched)
    avg_multiple_score = sum(matched_mult_high_scores ) / len(uncertain)
    print("The average score for exact matches is {}".format(avg_matched_score))
    print("The average highest score for multiple matches is {}".format(avg_multiple_score))

    # plot the matched scores in a histogram
    n, bins, patches = plt.hist(matched_scores, 10, facecolor="blue", alpha=0.5)
    plt.show()

    n, bins, patches = plt.hist(matched_mult_high_scores, 10, facecolor="blue", alpha=0.5)
    plt.show()

    with open("result_matched.csv", "w") as stats_file:
        wr = csv.writer(stats_file, dialect="excel")
        wr.writerow(["Score", "Template ID", "Template Version", "CondText"])
        for ID, match in matched.items():
            wr.writerow([str(x) for x in match] + [ orig_conditions[ID] ])

    with open("result_multi_matched.csv", "w") as stats_file:
        wr = csv.writer(stats_file, dialect="excel")
        wr.writerow(["Score", "Template ID", "Template Version", "CondText", "Match Rank"])
        for ID, matches in uncertain.items():
            for i, match in enumerate(matches):
                wr.writerow([str(x) for x in match] + [orig_conditions[ID], str(i+1)])
