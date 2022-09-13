import glob
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import os

def merge_csv(csv_name):
    
    list_csv = glob.glob("result/temp/split_metric_result/test_" + csv_name.split("_split")[0] +"*")
    if len(list_csv) > 0:
        dfs = []
        for csv in list_csv:
            df = pd.read_csv(csv)
            if len(df)<2:
                print(f"delte {csv}")
                
                if "_split-" in Path(csv).stem:
                    df["split"] = int(Path(csv).stem.split("_split-")[1].split("_")[0])
                Path(csv).unlink()
                df = dfs.append(df) 
        all_split = pd.concat(dfs, ignore_index=True)
        all_split.to_csv(path_or_buf="test_" + csv_name.split("_split")[0] + ".csv",index=False)
    elif len(list_csv) in [0,1]:
        print(f"size of csv is {len(list_csv)}")
    else:
        print("some csv are missing")
        breakpoint()

def merge_result(csv_name):
    print("to work on it")

    Path("result/temp/split_confusion_result").mkdir(parents=True, exist_ok=True)

    list_npy = glob.glob("result/temp/split_confusion_result/Confusion_" + csv_name.split("_result")[0] +"*")
    print(f"found {len(list_npy)} nunmpy file for confusion ")
    if len(list_npy) ==0:
        breakpoint()
    assert(len(list_npy) > 0)
    A = []
    for split in range(0,len(list_npy)+1):
        npy = glob.glob("result/temp/split_confusion_result/Confusion_" + csv_name.split("_result")[0] +f"*{split}.npy")
        if len(npy) != 1:
            A.append(np.load(list_npy[0]))
            continue
        A.append(np.load(npy[0]))

    cc = np.asarray(A).mean(axis=0)
    if cc.shape[0]>9:
        x_axis_labels = ['Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', 
                        'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil',
                        'Relief shadow','Water','Urbanized areas']
    else:
        x_axis_labels = ['Urbanized areas', 'Water', 'Forest', 'Moor', 'Orchards', 'Vineyards', 'Other crops']

    fi = sns.heatmap(cc, annot=True,fmt=".2f",cbar=False,xticklabels=x_axis_labels, yticklabels=x_axis_labels)
    fi.set_title(f"{Path(csv_name).stem}".split("_split")[0])

    Path("result/figure").mkdir(parents=True, exist_ok=True)
    s_ = str( Path("result") / "figure" / str(f"{Path(csv_name).stem}".split("_split")[0] + ".png"))
    # if os.path.isfile(s_):
    #     os.remove(s_)
    fi.figure.savefig(s_,bbox_inches = 'tight')
    fi.figure.clf()
    print(f"save fig {Path(s_).stem}")

# site = "data_dordone"
# method = "0612"
# split=0
# for sensor in ["S1"]:

#     csv_name = f"{method}_{sensor}_site-{site}_result_split-{split}.csv"
#     merge_csv(csv_name)
