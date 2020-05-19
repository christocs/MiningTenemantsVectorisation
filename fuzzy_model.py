import os, csv, time, re, collections
from fuzzywuzzy import fuzz, process

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
            result.append(Template(row_text, int(row['Identifier']), int(row['Version No'])))
        print("Imported {0} conditions templates in {1:.2f} seconds\n".format(len(result), time.time() - start_time))
        return result

def fuzzy_match(templates):
    start_time = time.time()
    ranks = []
    print("Matching strings")
    for template in templates:
        nearest = process.extract(template, templates)#, limit=len(templates))
        try:
            rank = [temp.id for temp, score in nearest].index(template.id)
        except ValueError:
            rank = None
        ranks.append(rank)
    print("{0} templates matched in {1:.2f} seconds\n".format(len(templates), time.time() - start_time))
    return ranks


if __name__ == '__main__':
    templates = read_templates(os.getcwd() + "/tenement_templates_dupes_removed.csv")

    ranks = fuzzy_match(templates)
    counter = collections.Counter(ranks)
    print(counter)
    
