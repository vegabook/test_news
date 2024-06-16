# test embeddings
import os
import sys
import pyarrow.csv 
import pyarrow as pa
import pandas as pd
import pathlib
import json
import IPython
import re
from bertopic import BERTopic
import numpy as np
import pickle
from openai import OpenAI
from bertopic.backend import OpenAIBackend
from bertopic.dimensionality import BaseDimensionalityReduction
import datetime as dt
import time
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline
from functools import reduce

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import Fore, Back, Style, init as colinit
from pprint import pprint
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)



DELPHOS_NEWS_CSV = "./data/dfi_news/dfi_pr.csv"
DELPHOS_TXT_FIELD = "text"
DELPHOS_TITLE_FIELD = "title"
JSON_KEY_LOCATION = "~/.ssh/openai_secret_key.json"
JSON_KEY_NAME = "SemTrend_key"


def get_key_json(jsonfile, keyname):
    jsonfile_exp = pathlib.Path(jsonfile).expanduser()
    with open(jsonfile_exp) as f:
        return json.load(f)[keyname]


def make_client(openai_key):
    key = get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME)
    # set environemnt variable
    os.environ["OPENAI_API_KEY"] = key
    return OpenAI() # will get key from environment


def get_datafile_csv(file_location):
    # expand file location for tilde
    file_location_exp = pathlib.Path(file_location).expanduser()
    reader = pyarrow.csv.open_csv(file_location_exp)
    return reader.read_pandas()


def prepare_data(data_table, field, window_days, window_stride, longchop):
    windows = []
    dates = pd.to_datetime(data_table["dfi_upload_timestamp"])
    data_table.set_index(dates, inplace = True)

    # make the windows
    if window_days > 0:
        start_date = list(dates)[0]
        end_date = list(dates)[-1]
        while start_date + pd.DateOffset(days = window_days) <= end_date:
            end_window = start_date + pd.DateOffset(days = window_days)
            windows.append(dates[(dates >= start_date) & (dates < end_window)])
            start_date = start_date + pd.DateOffset(days = window_stride)
    else:
        windows.append(dates)
    # split the data_table into data windows
    data_windows = []
    for window in windows:
        tf = np.isin(data_table.index.to_numpy(), window.to_numpy())
        data_windows.append(data_table.loc[tf, :])

    prep_windowed_data = []
    for data_window in data_windows:
        text = data_window[field].dropna()
        stripped = text.str.strip()
        clean_html_regex = re.compile("<.*?>")
        no_html = [re.sub(clean_html_regex, "", i) for i in stripped]
        no_carriage = [i.replace("\n", " ") for i in no_html]
        # no bigger than 5000 characters
        no_long = [i[:longchop] for i in no_carriage]
        # no less than 200 characters
        no_short = [i for i in no_long if len(i) > 100]
        prep_windowed_data.append(no_short)

    # now check that all windows have data and are unique
    for i in range(len(prep_windowed_data)):
        print(f"Window {i} has {len(prep_windowed_data[i])} articles", end = "  ")
    print()
    return prep_windowed_data


def do_bert(data):
    count = 0
    results = []
    for d in data:
        print(f"{count}/{len(data)} ", end = " ")
        sys.stdout.flush()
        topic_model = BERTopic()
        window_result = topic_model.fit_transform([d])
        results.append((topic_model, window_result))
        count += 1
    return results

def do_openai_embedding(batches, model, dimensions):
    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))
    topic_model = BERTopic(embedding_model = "openai-gpt") # english
    tokens_used = 0
    batches_done = 0
    nowtime = dt.datetime.now()
    for b in batches:
        batches_done += 1
        if len(b) > 0:
            try:
                e = client.embeddings.create(
                    input = b,
                    model = model,
                    dimensions = dimensions)
                yield (batches_done, e.data)
            except Exception as e:
                print(f"{Fore.RED}Error in batch {batches_done}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{e}{Style.RESET_ALL}")
                yield (batches_done, [None for _ in range(len(b))])
                continue
        else:
            print(f"{Fore.CYAN}batch {batches_done} has length 0 {Style.RESET_ALL}")
            yield (batches_done, [])
        tokens_used += e.usage.total_tokens
        time_taken = (dt.datetime.now() - nowtime).total_seconds()
        print(f"Done {batches_done} out of {len(batches)}, time taken is {time_taken}, tokens used is {tokens_used}")


def do_sentence_transformers():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    dphs = get_datafile_csv(DELPHOS_NEWS_CSV)
    data = prepare_data(dphs, field = DELPHOS_TXT_FIELD, window_days = 0, window_stride = 0)[0]
    embeddings = embedding_model.encode(data, show_progress_bar=True)
    return embeddings


def do_language_detection(load = False):
    if load:
        with open("./output/languages/languages.pickle", "rb") as p:
            return pickle.load(p)
    dphs = get_datafile_csv(DELPHOS_NEWS_CSV)
    data = prepare_data(dphs, field = DELPHOS_TXT_FIELD, window_days = 0, window_stride = 0)[0]
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt)
    batchsize = 20
    batches = [data[i:(i + batchsize)] for i in range(0, len(data), batchsize)]
    results = []
    numdone = 0
    nowtime = dt.datetime.now()
    for batch in batches:
        langs = pipe(batch, top_k=2, truncation=True)
        results.extend(langs)
        for i in range(len(langs)):
            x = langs[i]
            if x[0]["label"] != "en":
                print(x)
            if x[0]["score"] < 0.8:
                print(x)
                print(Fore.RED, Style.BRIGHT, f"Low score {x[0]['score']}", Style.RESET_ALL)
                print(Fore.YELLOW, batch[i], Style.RESET_ALL)
                time.sleep(5)


        numdone += len(batch)
        seconds = (dt.datetime.now() - nowtime).total_seconds()
        print(f"Done {numdone} out of {len(data)} time is {dt.datetime.now()} seconds is {seconds}")
    return results

def submit_completion_batch(
        data, 
        batchname = "english_3_sentences",
        model = "gpt-3.5-turbo",
        role_content = ("You will be provided with a document, "
                       "your task is to summarise it succinctly "
                       "in English using a maximum of 4 sentences."),
        batchindices = None):

    logger.info(f"Starting batch {batchname}")
    if batchindices is not None:
        data = data[batchindices[0]:batchindices[1]]
    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))
    logger.info(f"Creating batch")
    count = 0
    batch = []
    for datum in data:
        thisline = {
                "custom_id": f"request-{count:0>6}",
                "method": "POST", 
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages":[
                                {
                                  "role": "system",
                                  "content": role_content, 
                                },
                                {
                                  "role": "user",
                                  "content": f"{datum}"
                                }
                              ],
                    "temperature": 0.7,
                    }
                }
        batch.append(thisline)
        count += 1

    logger.info(f"Saving batch")
    # now save json
    with open(f"./output/{batchname}.json", "w") as f:
        for j in batch:
            f.write(f"{json.dumps(j)}\n")
    logger.info(f"Uploading batch")
    batch_input_file = client.files.create(
            file = open(f"./output/{batchname}.json", "rb"),
            purpose = "batch"
            )
    batch_input_file_id = batch_input_file.id
    # now upload to openai batch api
    batch_object = client.batches.create(
            input_file_id = batch_input_file_id,
            endpoint = "/v1/chat/completions",      
            completion_window = "24h",
            metadata = {"description": batchname}
            )
    return batch_object

def check_completion_batch(batchid):
    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))
    return client.batches.retrieve(batchid)

def list_batches():
    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))
    return [(b.id, b.status, b.metadata, b.request_counts) 
            for b in list(client.batches.list())]


def do_tfidf():
    dphs = get_datafile_csv(DELPHOS_NEWS_CSV)
    data = prepare_data(dphs, field = DELPHOS_TXT_FIELD, window_days = 0, window_stride = 0)[0]
    vectorizer = TfidfVectorizer()
    results = vectorizer.fit_transform(data, progress_bar = True)
    return results 


def get_pickle_files():
    files = os.listdir("./output")
    for f in files:
        with open(f"./output/{f}", "rb") as p:
            data = pickle.load(p)
            breakpoint()

def get_openai_batch_results(
    filters = [lambda x: "francais_trois_phrases" in x[0].lower()]
    ):
    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))

    # apply filters
    x0 = [(b.metadata["description"], b.output_file_id) 
          for b in client.batches.list() if b.status == "completed"]
    xf = list(x0)

    # filter
    for f in filters:
        print(f"{len(xf)=}")
        xf = list(filter(f, xf))
    xs = sorted(xf, key = lambda x: x[0])
    labels = [x[0] for x in xs]

    # check for duplicates 
    if len(set(labels)) != len(labels):
        print("Warning, duplicate labels, removing")
        newlist = []
        for x in xs:
            # On**2 but who cares 
            if x[0] not in [n[0] for n in newlist]:
                newlist.append(x)
        xs = newlist
    print(f"{len(xs)=}")

    # combine
    cont = [client.files.content(x[1]) for x in xs]

    # separate
    sepped = [json.loads(x)["response"]["body"]["choices"][0]["message"]["content"] 
              for c in cont 
              for x in c.iter_lines()]
    return sepped


def make_openai_batches(data, batchname, model, role_content, batchsize):
    """ example: 
        fr = make_openai_batches(data, 
                             batchname = 'Francais_trois_phrases_gpt4o', 
                             model = 'gpt-4o', 
                             role_content = 'Traduisez en français')
        note data must be a list of text strings """

    client = make_client(get_key_json(JSON_KEY_LOCATION, JSON_KEY_NAME))
    indices = list(range(0, len(data), batchsize)) + [len(data)]
    for i0, i1 in zip(indices[:-1], indices[1:]):
        is_submitted = False
        while not is_submitted:
            ibatchname = f"{batchname}_{model}_i{i0:0>6}_i{i1:0>6}"
            print(f"Submitting batch {ibatchname}")
            batch_object = submit_completion_batch(
                    data,
                    batchname = ibatchname,
                    model = model,
                    role_content = role_content,
                    batchindices = (i0, i1))
            pprint(list(batch_object))
            time.sleep(30)
            xx = client.batches.retrieve(batch_object.id)
            print(f"status is {xx.status}")
            pprint(xx)
            while xx.status not in ["in_progress", "failed"]:
                time.sleep(10)
                xx = client.batches.retrieve(batch_object.id)
                print(f"status is {xx.status}")
            if xx.status == "in_progress":
                is_submitted = True
            elif xx.status == "failed":
                time.sleep(10)

        while xx.status != "completed":
            time.sleep(10)
            xx = client.batches.retrieve(batch_object.id)
    print("done, might want to download batches now")

if __name__ == "__main__":
    do_translation = True
    model = "gpt-3.5-turbo"
    batchsize = 15000
    if do_translation: 
        data = prepare_data(get_datafile_csv(DELPHOS_NEWS_CSV),
                            field = DELPHOS_TXT_FIELD, 
                            window_days = 0, 
                            window_stride = 0, 
                            longchop = 100000)[0]
        data = prepare_data(dphs, field = DELPHOS_TXT_FIELD, 
                            window_days = 0, 
                            window_stride = 0, 
                            longchop = 1000)[0]
        batchname = "france_test_7"
        fr = make_openai_batches(data, 
                                 batchname = batchname,
                                 model = model,
                                 role_content = "Vous serez fourni avec un document. Votre tâche est de le traduire français.", 
                                 batchsize = batchsize)

        s = get_openai_batch_results(filters = [lambda x: batchname in x[0]])
        with open(f"./output/{batchname}-{model}.pickle", "wb") as f:
            pickle.dump(s, f)

        dphs = get_datafile_csv(DELPHOS_NEWS_CSV)
        data = prepare_data(dphs, field = DELPHOS_TXT_FIELD, 
                            window_days = 0, 
                            window_stride = 0, 
                            longchop = 1000)[0]
        batchname = "england_test_7"
        fr = make_openai_batches(data, 
                                 batchname = batchname,
                                 model = model,
                                 role_content = "You will be given a document. Your task is to translate it into English.", 
                                 batchsize = batchsize) 

        s = get_openai_batch_results(filters = [lambda x: batchname in x[0]])
        with open(f"./output/{batchname}-{model}.pickle", "wb") as f:
            pickle.dump(s, f)


    do_embeddings = True
    embedding_model = "text-embedding-3-small"
    dimensions = 1024
    batchsize = 1000
    if do_embeddings: 
        # full article embeddings
        for batchname in [f"france_test_7-{model}", f"england_test_7-{model}"]:
            data = pd.read_pickle(f"./output/{batchname}.pickle")

            # sentence embeddings
            batches = [[x for x in a.split(".") if x!= ""] for a in data]
            for idx, result in do_openai_embedding(batches = batches, 
                                                   model = embedding_model, 
                                                   dimensions = dimensions):
                with open(f"./output/{batchname}-{model}-{idx:06}--sentences--{embedding_model}-{dimensions}.pickle", "wb") as f:
                    pickle.dump(result, f)

            # full article embeddings
            batches = [data[i:(i + batchsize)] for i in range(0, len(data), batchsize)]
            for idx, result in do_openai_embedding(batches = batches, 
                                                   model = embedding_model, 
                                                   dimensions = dimensions):
                with open(f"./output/{batchname}-{model}-{idx:06}--{embedding_model}-{dimensions}.pickle", "wb") as f:
                                pickle.dump(result, f)


