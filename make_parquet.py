import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import pickle
from test_news import DELPHOS_NEWS_CSV, DELPHOS_TXT_FIELD,\
    get_datafile_csv, prepare_data
import numpy as np
import random
from colorama import Fore, Back, Style, init as colinit
colinit()
from pprint import pprint

def path_filter(path: str, includes: list, excludes: list) -> list:
    must = [p for p in Path(path).rglob('*.pickle') 
        if all(i in str(p) for i in includes)]
    mustnt = [p for p in must if not any(e in str(p) for e in excludes)]
    return mustnt

def read_pickle(path: str) -> pa.Table:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    output_dir = Path("/home/tbrowne/data/test_news")
    # make if not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # first raw data
    rawdata = prepare_data(get_datafile_csv(DELPHOS_NEWS_CSV),
                        field = DELPHOS_TXT_FIELD, 
                        window_days = 0, 
                        window_stride = 0, 
                        longchop = 10000000)[0]
    outdir = output_dir / "rawdata"
    outdir.mkdir(parents=True, exist_ok=True)
    # erase everything in that directory
    print(Fore.RED + Style.BRIGHT + f"Erasing {outdir}", Style.RESET_ALL)
    for f in outdir.glob("*"):
        f.unlink()

    count = 0
    for data in rawdata:
        table = pa.table({f"{count:06d}": pa.array(data)})
        # write the table
        pq.write_table(table, outdir / f"{count:06d}.parquet")
        print(Fore.CYAN + f"Written {count:06d}.parquet to {outdir}", Style.RESET_ALL)
        count += 1

    # now processed data
    sets = {
            #("FR", "SENTENCE_EMBEDDING", "GPT3.5"): ["/home/tbrowne/code/test_news/output", 
            #            ["france_test_7", "sentences", "embedding", "small"], []],
            #("EN", "SENTENCE_EMBEDDING", "GPT3.5"): ["/home/tbrowne/code/test_news/output", 
            #            ["england_test_7", "sentences", "embedding", "small"], []],
            #("FR", "ARTICLE_EMBEDDING", "GPT3.5"): ["/home/tbrowne/code/test_news/output",
            #            ["france_test_7", "embedding", "small"], ["sentences"]],
            #("EN", "ARTICLE_EMBEDDING", "GPT3.5"): ["/home/tbrowne/code/test_news/output",
            #            ["england_test_7", "embedding", "small"], ["sentences"]],
            #("FR", "ARTICLE_TRANSLATION", "GPT3.5"): ["/home/tbrowne/code/test_news/output",
            #            ["france_test_7"], ["sentences", "embedding"]],
            #("EN", "ARTICLE_TRANSLATION", "GPT3.5"): ["/home/tbrowne/code/test_news/output",
            #            ["england_test_7"], ["sentences", "embedding"]],


            ("FR", "SENTENCE_EMBEDDING", "GPT4o"): ["/home/tbrowne/code/test_news/output0", 
                        ["france_test_2", "sentences", "embedding", "large"], []],
            ("EN", "SENTENCE_EMBEDDING", "GPT4o"): ["/home/tbrowne/code/test_news/output0",
                        ["england_test_2", "sentences", "embedding", "large"], []],
            ("FR", "ARTICLE_EMBEDDING", "GPT4o"): ["/home/tbrowne/code/test_news/output0",
                        ["france_test_2", "embedding", "large"], ["sentences"]],
            ("EN", "ARTICLE_EMBEDDING", "GPT4o"): ["/home/tbrowne/code/test_news/output0",
                        ["england_test_2", "embedding", "large"], ["sentences"]],
            ("FR", "ARTICLE_TRANSLATION", "GPT4o"): ["/home/tbrowne/code/test_news/output0",
                        ["france_test_2"], ["sentences", "embedding"]],
            ("EN", "ARTICLE_TRANSLATION", "GPT4o"): ["/home/tbrowne/code/test_news/output0",
                        ["england_test_2"], ["sentences", "embedding"]]
    }

    for k in sets.keys():
        v = sets[k]
        print(Fore.GREEN + f"Processing {k} {v}", Style.RESET_ALL)
        files = path_filter(v[0], v[1], v[2])
        files = sorted(files)
        print(Fore.YELLOW, f"Found {len(files)} files", Style.RESET_ALL)
        if len(files) < 10:
            pprint(files)
        else:
            pprint(random.sample(files, 10))


        # make the directory
        outdir = output_dir / "_".join(k)
        outdir.mkdir(parents=True, exist_ok=True)
        # erase everything in that directory
        print(Fore.RED + Style.BRIGHT + f"Erasing {outdir}", Style.RESET_ALL)
        for f in outdir.glob("*"):
            f.unlink()

        if k[1] == "SENTENCE_EMBEDDING":
            # read the pickle files
            count = 0
            for f in files:
                data = np.array([np.array(x.embedding) 
                    if x is not None else np.array([np.nan]) for x in read_pickle(f)])
                table = pa.Table.from_arrays(data, 
                                             names=[f"{count:06d}-{i:03d}" for i in range(len(data))])
                pq.write_table(table, outdir / f"{count:06d}.parquet")
                print(Fore.CYAN + f"Written {count:06d}.parquet to {outdir}", Style.RESET_ALL)
                count += 1
        elif k[1] == "ARTICLE_EMBEDDING":
            # read the pickle files
            count = 0
            for f in files:
                alldata = np.array([np.array(x.embedding) 
                    if x is not None else np.array([np.nan]) for x in read_pickle(f)])
                for data in alldata:
                    # write numpy to parquet
                    table = pa.table({f"{count:06d}": pa.array(data)})
                    # write the table
                    pq.write_table(table, outdir / f"{count:06d}.parquet")
                    print(Fore.CYAN + f"Written {count:06d}.parquet to {outdir}", Style.RESET_ALL)
                    count += 1
        elif k[1] == "ARTICLE_TRANSLATION":
            # read the pickle files
            count = 0
            for f in files:
                alldata = np.array(read_pickle(f))
                for data in alldata:
                    # make the table
                    table = pa.table({f"{count:06d}": pa.array(data)})
                    # write the table
                    pq.write_table(table, outdir / f"{count:06d}.parquet")
                    print(Fore.CYAN + f"Written {count:06d}.parquet to {outdir}", Style.RESET_ALL)
                    count += 1














