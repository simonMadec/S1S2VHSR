import glob
import pandas as pd
import numpy as np
from pathlib import Path


def merge_csv(csv_name):
    list_csv = glob.glob("test_" + csv_name.split("_result")[0] +"*")

    if len(list_csv) > 4:
        dfs = []
        for csv in list_csv:
            df = pd.read_csv(csv)
            if len(df)<2:
                print(f"delte {csv}")
                Path(csv).unlink()
                df["split"] = int(Path(csv).stem.split("_split-")[1])
                df = dfs.append(df) 
        all_split = pd.concat(dfs, ignore_index=True)
        all_split.to_csv(path_or_buf="test_" + csv_name.split("_split")[0] + ".csv",index=False)

    elif len(list_csv) in [0,1]:
        print(f"size of csv is {len(list_csv)}")
        breakpoint()

    else:
        print("some csv are missing")
        breakpoint()

# site = "data_dordone"
# method = "0612"
# split=0
# for sensor in ["S1"]:

#     csv_name = f"{method}_{sensor}_site-{site}_result_split-{split}.csv"
#     merge_csv(csv_name)
