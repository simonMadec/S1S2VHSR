from pathlib import Path
import rasterio
import glob
import numpy as np

for x in glob.glob("/home/simon/DATA/land_use_classification/data_ToyDordogne/Sentinel-2/*tif"):
    with rasterio.open(x) as src:
        print(f"reading.. {Path(x).stem}")
        ts_array = src.read()
        print("Min Max Standardisation")
        print(Path(x).stem)
        print(np.unique(ts_array))