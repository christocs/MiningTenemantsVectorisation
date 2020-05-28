import os, time, csv, re, random, pickle, collections
import gensim
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import matplotlib.mlab as mlab
from matplotlib import colors
import matplotlib.pyplot as plt
from tabulate import tabulate

NEAR_MATCH_THRESHOLD = 0.03 # how close a template match must be to the highest match to be considered significant
EPOCHS = 10

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


def vector_match(model, templates, conditions = [], multimatch = True): 
    """
    Matches conditions to templates, using the pretrained (doc2vec) model.

    If multimatch is true, the function will match > 1 template if there are templates
    within NEAR_MATCH_THRESHOLD of the highest match score AND if the template ID
    differs from all other near matches (i.e. top matches with unique template IDs)

    If no conditions are supplied the templates will be matched against themselves.

    returns 2 lists: 
    - single matches (all conditions if multimatch = False)
    - multiple matches (empty if multimatch = False)
    """
    print("Matching the documents")
    start_time = time.time()

    # conditions is expected to be a list of list of words
    if len(conditions) == 0: # if no conditions are supplied, just match templates against themselves
        conditions = [gensim.utils.simple_preprocess(" ".join(t.words)) for t in templates]
    single_matches = {}
    multiple_matches = {}
    for cID, condition in enumerate(conditions):
        condition_vector = model.infer_vector(condition)
        most_similar = model.docvecs.most_similar([condition_vector], topn=10) 
        highest, highest_score  = ( templates[most_similar[0][0]], most_similar[0][1] )

        if not multimatch: # just find the highest match
            single_matches[cID] = (highest_score, highest.id, highest.version)
        else:
            # determine how many other templates rank close to this one
            # only consider DIFFERENT TEMPLATE IDs, not different versions of the same ID
            similar_template_ids = [ highest.id ] # start with best match
            num_close_matches = 0                 # and count how many other IDs are nearby
            for tID, score in most_similar:
                current = templates[tID]
                if abs(highest_score - score) < NEAR_MATCH_THRESHOLD and current.id not in similar_template_ids:
                    num_close_matches = num_close_matches + 1
            
            if num_close_matches == 0:
                single_matches[cID] = (highest_score, highest.id, highest.version)
            else:
                multiple_matches[cID] = [(sim[1], templates[sim[0]].id, templates[sim[0]].version) for sim in most_similar[0:num_close_matches+1]]
    print("{0} documents compared in {1:.2f} seconds\n".format(len(conditions), time.time() - start_time))
    return single_matches, multiple_matches


def multimatch_summary(multimatched):
    ranks = []
    for ID, matches in multimatched.items():
        ranks.append(len(multimatched))
    counter = collections.Counter(ranks)
    print("The number of matches per uncertain document are:")
    print(counter)


def score_summary(scores):
    if len(scores) == 0:
        return
    scores.sort()
    half = len(scores) // 2
    quart = half // 2
    header = ["Min", "1st Quartile", "Median", "Mean", "3rd Quartile", "Max"]
    stats = [ min(scores), scores[quart], np.median(scores), np.mean(scores), scores[half+quart], max(scores) ]
    print(tabulate([stats], headers=header))

def frequency_plot(data):
    if len(data) == 0:
        return
    n, bins, patches = plt.hist(data, bins=10)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    
    # colour code by height
    fracs = n / n.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    plt.show()


if __name__ == '__main__':
    templates = read_templates(os.getcwd() + "/tenement_templates_regex.csv")
    orig_conditions = read_conditions(os.getcwd() + "/ConditionsNoRegexMatches.csv", "CondText")
    conditions = tokenise(orig_conditions)
            
    model = train_model(templates)

    # Assess the model with the training data (sanity check)
    matched, uncertain = vector_match(model, templates, multimatch=False)
    #  matched, uncertain = vector_match(model, templates, conditions)
    print("{0} documents have been paired to a template ID exactly, {1} have multiple matches\n".format(len(matched), len(uncertain)))
    
    multimatch_summary(uncertain)

    # Descriptive statistics
    matched_scores = [m[0] for k,m in matched.items()]
    matched_mult_high_scores = [u[0][0] for k,u in uncertain.items()]
    print("\nMatched Scores:")
    score_summary(matched_scores)
    print("\nMulti-Matched High Scores:")
    score_summary(matched_mult_high_scores)

    # plot the matched scores in a histogram
    frequency_plot(matched_scores)
    frequency_plot(matched_mult_high_scores)

    # write the results to csv files
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
