import os, csv, time, re, collections
from fuzzywuzzy import fuzz, process
from gensim.utils import simple_preprocess
from doc2vec_model import read_conditions, frequency_plot, score_summary

# A class that inherits from the built-in string and adds a couple of
# useful attributes for later comparison.
class Template(str):
    def __new__(self, value, id, version):
        obj = str.__new__(self, value)
        obj.id = id
        obj.version = version
        return obj


def read_templates(fname, tokens_only=False):
    start_time = time.time()
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result = []
        for row in reader:
            row_text = re.sub("\*\*.*\*\*('s)?", "", row['Text'])
            row_text = " ".join(simple_preprocess(row_text))
            result.append(Template(row_text, int(row['Identifier']), int(row['Version No'])))
        print("Imported {0} conditions templates in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
        return result

def fuzzy_match(templates, conditions):
    start_time = time.time()
    result = []
    print("Matching strings")
    for cond in conditions:
        template, score = process.extractOne(cond, templates, scorer=fuzz.token_set_ratio)
        result.append([ score, template.id, template.version ])
    print("{0} conditions matched in {1:.2f} seconds\n".format(len(conditions), time.time() - start_time))
    return result


if __name__ == '__main__':
    templates = read_templates(os.getcwd() +  "/tenement_templates_regex_no_overlapping_templates.csv")
    orig_conditions = read_conditions(os.getcwd() + "/error_calculation/ConditionsWithRegexMatchesThatOnlySatisfyGreedyRegex.csv", "CondText")

    matched = fuzzy_match(templates, orig_conditions)
    matched_scores = [ score for score, i, v in matched ]
    score_summary(matched_scores)
    frequency_plot(matched_scores)
    
    # write the results to csv files
    with open("fuzzy_result_matched.csv", "w") as stats_file:
        wr = csv.writer(stats_file, dialect="excel")
        wr.writerow(["Score", "Template ID", "Template Version", "CondText"])
        for i, match in enumerate(matched):
            wr.writerow([str(x) for x in match] + [ orig_conditions[i] ])

    print() # print a new line only
