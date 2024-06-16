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
import numpy as np
import pickle
import datetime as dt
import time
import cohere
import IPython

from colorama import Fore, Back, Style, init as colinit
from pprint import pprint
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

COHERE_JSON_KEY_LOCATION = "~/.ssh/cohere_secret_key.json"
COHERE_JSON_KEY_NAME = "Cohere_main"

DELPHOS_NEWS_CSV = "./data/dfi_news/dfi_pr.csv"
DELPHOS_TXT_FIELD = "text"
DELPHOS_TITLE_FIELD = "title"
DELPHOS_DATE_FIELD = "dfi_upload_timestamp"

def get_key_json(jsonfile, keyname):
    jsonfile_exp = pathlib.Path(jsonfile).expanduser()
    with open(jsonfile_exp) as f:
        return json.load(f)[keyname]


def make_cohere_client(cohere_key = get_key_json(COHERE_JSON_KEY_LOCATION, COHERE_JSON_KEY_NAME)):
    key = get_key_json(COHERE_JSON_KEY_LOCATION, COHERE_JSON_KEY_NAME)
    # set environemnt variable
    co = cohere.Client(api_key = key)
    return co

def get_datafile_csv(file_location = DELPHOS_NEWS_CSV):
    # expand file location for tilde
    file_location_exp = pathlib.Path(file_location).expanduser()
    reader = pyarrow.csv.open_csv(file_location_exp)
    return reader.read_pandas()


def prepare_data(data_table = get_datafile_csv(), 
                 field = DELPHOS_TXT_FIELD, 
                 timestamp_field = DELPHOS_DATE_FIELD, 
                 window_days = 0, 
                 window_stride = 7):
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
        subset = (data_window[[field, timestamp_field]]).dropna(how = "any")
        stamp = subset[timestamp_field].to_list()
        text = subset[field].to_list()
        stripped = [s.strip() for s in text]
        clean_html_regex = re.compile("<.*?>")
        no_html = [re.sub(clean_html_regex, "", i) for i in stripped]
        no_carriage = [i.replace("\n", " ") for i in no_html]
        # no bigger than 5000 characters
        no_long = [i[:2000] for i in no_carriage]
        # no less than 200 characters
        no_short = [(i[0], i[1]) for i in zip(no_long, stamp) if len(i[0]) > 100]
        prep_windowed_data.append(no_short)

    # now check that all windows have data and are unique
    for i in range(len(prep_windowed_data)):
        print(f"Window {i} has {len(prep_windowed_data[i])} articles", end = "  ")
    print()
    return prep_windowed_data

if __name__ == "__main__":
    co = make_cohere_client()
    data = prepare_data(get_datafile_csv(DELPHOS_NEWS_CSV), field = DELPHOS_TXT_FIELD, window_days = 0, window_stride = 0)[0]
    count = 1030
    thiscount = 0
    fulltime = dt.datetime.now()
    fails = {}
    output_directory = "./output/cohere_jp_de/"
    already_have = [int(x.split("_")[1].split(".")[0]) for x in os.listdir(output_directory)]
    need_to_do = sorted(list(set(list(range(len(data)))) - set(already_have)))
    skip = [22577, 1030]
    for di in need_to_do:
        d = data[di][:1030]
        done = False
        while not done:
            try:
                if di in skip:
                    en_tran = ""
                    fr_tran = ""
                else:
                    nowtime = dt.datetime.now()
                    print(Fore.YELLOW, count, d[1], Fore.RESET)
                    nowtime = dt.datetime.now()
                    message = f"以下の文書を日本語に翻訳し、最大4文で要約してください。翻訳のみを返してください: {d[0]}"
                    print(message)
                    response = co.chat(
                            model = "command-r",
                            message = message)
                    en_tran = response.text
                    print(Fore.RED, en_tran, Fore.RESET)

                    message = f"Übersetzen Sie das folgende Dokument ins Deutsche und fassen Sie es in maximal 4 Sätzen zusammen. Geben Sie nur die Übersetzung zurück: {d[0]}"
                    response = co.chat(
                            model = "command-r",
                            message = message)
                    fr_tran = response.text
                    print(Fore.BLUE, fr_tran, Fore.RESET)

                en_embed = co.embed(texts = [en_tran], 
                                    model = "embed-multilingual-v3.0",
                                    input_type = "classification").embeddings[0]

                fr_embed = co.embed(texts = [fr_tran], 
                                    model = "embed-multilingual-v3.0",
                                    input_type = "classification").embeddings[0]

                print((dt.datetime.now() - nowtime).total_seconds())

                fulldict = {"original": d[0], "timestamp": d[1],
                            "japanese": en_tran,
                            "german": fr_tran,
                            "japanese_embed": en_embed,
                            "german_embed": fr_embed}
                with open(f"{output_directory}cohere_{di:0>6}_japanese_german.pkl", "wb") as f:
                    pickle.dump(fulldict, f)
                thiscount += 1
                print((f"Done with {di} in {(dt.datetime.now() - fulltime).total_seconds()}"
                       f"at rate of {(dt.datetime.now() - fulltime).total_seconds() / thiscount}"))
                done = True
            except Exception as e:
                fails[dt.datetime.now()] = (count, d, e)
                print(Fore.RED, f"Failed at {count} with {e}...sleeping 5", Fore.RESET)
                time.sleep(5)

    print("all done")


    IPython.embed()
