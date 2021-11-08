


import glob
import pandas as pd
import numpy as np

sensor = ["Spot","S2"]
sensor = ["Sentinel-2"]
site = ["dordogne","reunion"]

JEacc = {"Spot" : { "dordogne": {"acc_mean" : 81.39, "acc_std" : 2.62}, "reunion" : {"acc_mean" : 88.35, "acc_std" : 1.33}}, "Sentinel-2" : { "dordogne": {"acc_mean" : 85.97, "acc_std" :  2.15}, "reunion" : {"acc_mean" : 87.98, "acc_std" : 1.12}}}

dfi = pd.DataFrame(columns=["Sensor","Acc_Py","Acc_TF","std_Acc_Py","std_Acc_TF"])

for si in site:
    for s in sensor:
        print(s)
        acc_all = []
        for split in range(0,5):
            list_csv = glob.glob(f"test_{s}*{si}*{split}.csv")

            if len(list_csv) == 0:
                print(f"missing results for site {si} sensor {s} split {split}")
                breakpoint()
            acc= []
            for csv in list_csv:
                print(csv)
                df = pd.read_csv(csv)
                acc.append(df["test_acc"].values[0])
            
            acc_all.append(np.mean(acc))
            if np.std(acc)>0.1:
                print(f"carefull high variation between repetition for site {si} sensor {s} split {split} ")
        
        new_row = {"Dataset": si,"Sensor": s, "Acc_Py": np.mean(acc_all)*100, "Acc_TF": JEacc[s][si]["acc_mean"], "std_Acc_Py": np.std(acc_all)*100, "std_Acc_TF": JEacc[s][si]["acc_std"]}
        dfi = dfi.append(new_row, ignore_index=True)
print(dfi)
dfi.to_csv(path_or_buf="GlobalComparaison_Py_TF.csv",index=False)
