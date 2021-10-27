
import pandas as pd 
import glob
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np

pathCSV = "results/spot"

for i in glob.glob( pathCSV + "/result*" ):
    plt.close()
    split = Path(i).stem.split("-")[1]
    df = pd.read_csv(i)

    plt.plot(np.array(df["epoch"].tolist()), np.array(df["val_acc"].tolist()),'--.',color='magenta',label="val_acc  split " + split)
    plt.plot(np.array(df["epoch"].tolist()), np.array(df["train_acc"].tolist()),'--.',color='blue',label="train_acc  split " + split)
    plt.xlabel("epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.ylim(0.5,1)
    plt.savefig(f"result{split}.png")

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
plt.savefig(f"resultglobal.png")