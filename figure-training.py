
import pandas as pd 
import glob
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

list_class=['Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', \
            'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil','Relief shadow','Water','Urbanized areas']

csv = glob.glob("perClasse-valid_Spot_result_split-all.csv")

for cs in csv: 
    df = pd.read_csv(cs)
    plt.figure(figsize=(20,5))
    sns.boxplot(data=df)
    plt.show()

    plt.xlabel("Classes")
    plt.ylabel("Acc")
    plt.savefig(f"spotAllClasses.png")
    breakpoint()
        

pathCSV = "results/S2"

for i in glob.glob( pathCSV + "/result*" ):
    plt.close()
    breakpoint()
    split = Path(i).stem.split("-")[-1]
    df = pd.read_csv(i)

    plt.plot(np.array(df["epoch"].tolist()), np.array(df["val_acc"].tolist()),'--.',color='magenta',label="val_acc  split " + split)
    plt.plot(np.array(df["epoch"].tolist()), np.array(df["train_acc"].tolist()),'--.',color='blue',label="train_acc  split " + split)
    plt.xlabel("epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.ylim(0.5,1)
    plt.savefig(f"S2result{split}.png")

plt.close()
for i in glob.glob( pathCSV + "/result*" ):
    
    split = Path(i).stem.split("-")[1]
    df = pd.read_csv(i)

    plt.plot(np.array(df["epoch"].tolist()), np.array(df["val_acc"].tolist()),'--.',color='magenta',label="val_acc ")
    plt.plot(np.array(df["epoch"].tolist()), np.array(df["train_acc"].tolist()),'--.',color='blue',label="train_acc")
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.legend("train_acc","val_acc")
plt.ylim(0.5,1)
plt.savefig(f"S2resultglobal.png")