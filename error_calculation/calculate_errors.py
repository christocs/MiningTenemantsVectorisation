import os, time, csv, collections
from tabulate import tabulate

def read(fname):
    start_time = time.time()
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            result.append(row)
    print("Imported {0} conditions from {1} in {2:.2f} seconds\n".format(len(result), os.path.basename(fname), time.time() - start_time))
    return result

def compare(bible, to_compare):
    a_name, b_name = to_compare.keys()
    a = to_compare[a_name]
    b = to_compare[b_name]
    a_report = [0, 0]
    b_report = [0, 0]
    for i, condition in enumerate(bible):
        if int(condition["RegexId"]) == int(a[i]["Template ID"]):
            a_report[0] = a_report[0] + 1
        else:
            a_report[1] = a_report[1] + 1

        if int(condition["RegexId"]) == int(b[i]["Template ID"]):
            b_report[0] = b_report[0] + 1
        else:
            b_report[1] = b_report[1] + 1
            
    a_report.append(a_report[0] * 100 / len(bible))
    b_report.append(b_report[0] * 100 / len(bible))
    a_report.insert(0, a_name)
    b_report.insert(0, b_name)
    return [ a_report, b_report ]

if __name__ == '__main__':
    reg = read(os.getcwd() + "/ConditionsWithRegexMatchesThatOnlySatisfyGreedyRegex.csv")
    d2v = read(os.getcwd() + "/Doc2VecOnGreedyRegex.csv")
    fuzz = read(os.getcwd() +  "/Doc2VecOnGreedyRegex.csv") # should be fuzzywuzzy output

    comparisons = compare(reg, {"doc2vec": d2v, "fuzzywuzzy": fuzz})
    header = ["Method", "Correct", "Incorrect", "Percentage Correct"]

    print(tabulate(comparisons, headers=header))
    print() # blank line at end of output
