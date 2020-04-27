import sys, os, csv, time, tempfile
from gensim.test.utils import datapath
from gensim import utils
import gensim.models

def getConditionsText(filename):
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row['CondText'].lower().split() for row in reader]
            
def save_model(model):
    with tempfile.NamedTemporaryFile(prefix='gensim-model-', dir=os.getcwd(), delete=False) as tmp:
        print(tmp.name)
        temporary_filepath = tmp.name
        model.save(temporary_filepath)

def load_model(filename):
    return gensim.models.Word2Vec.load(os.getcwd() + "/" + filename)

# If you run without any commandline args it will load Conditions.csv, build a model
# and save the model in a temp file in the same dir.
# If you run with a single argument (a model file name in the same dir) it will load the 
# previously built model.
if __name__ == '__main__':
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
    else:
        print("Importing data")
        start_time = time.time()
        conditions = getConditionsText(os.getcwd() + "/Conditions.csv")
        print("Imported {0} conditions in {1:.2f} seconds\n".format(len(conditions), time.time() - start_time))


        # build the model
        print("Building the model")
        start_time = time.time()
        model = gensim.models.Word2Vec(sentences=conditions, workers=2)
        print("Model build in {0:.2f} seconds\n".format(time.time() - start_time))
        
        save_model(model)

    print(model.wv.most_similar(positive=["aboriginal"], topn=5))
    print(model.wv.most_similar(positive=["waste"], topn=5))
    print(model.wv.most_similar(positive=["mike"], topn=5))

    for i, word in enumerate(model.wv.vocab):
        if i == 10:
            break
        print(word)
