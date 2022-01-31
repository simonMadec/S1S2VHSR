
import rasterio
import numpy as np
from tqdm import tqdm
from typing import List
import time
import torch
root = "/home/simon/DATA/land_use_classification" 
dataset = "data_reunion"

# read a raster tif file -> A/B from Spot/S12
SpotP_list_path = [f"{root}/{dataset}/Spot-P/2017_mosa_S6_P_TOA_ortho_kalideos_v4.tif"]
SpotMs_list_path = [f"{root}/{dataset}/Spot-MS/2017_mosa_S6_MS_TOA_ortho_kalideos_v4.tif"]
S1_list_path = [f"{root}/{dataset}/Sentinel-1/{x}_ASC_CONCAT_S1.tif" for x in ["VV","VH"]]
S2_list_path =  [f"{root}/{dataset}/Sentinel-2/{x}_REUNION_2017_CONCAT_S2_GAPF.tif" for x in ["B2","B3","B4","B8","NDVI","NDWI"]] 

def get_image_latlong(raster_path: str)-> (List[float], List[float]):
    with rasterio.open(raster_path) as src:
        band1 = src.read(1)
        print('Band1 has shape', band1.shape)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons= np.array(xs)
        lats = np.array(ys)
        print('lons shape', lons.shape)
        return lats, lons

def latlong2pixels(raster_src, lat, lon):
    py, px = raster_src.index(lon, lat)
    return py, px

def read_sensor(input: List):
    array = []
    for i,x in enumerate(input):
        with rasterio.open(input[i]) as src:
            ts_array = src.read()
            ts_array = np.moveaxis(ts_array,0,-1)
            array.append(ts_array)
        ts_array=None

    if "Sentinel-1" in x:
        return np.moveaxis(np.dstack(array),-1,0),src
    elif "Spot-MS" in x:
        return np.moveaxis(np.squeeze(np.asarray(array),axis=0),-1,0),src
    elif "Sentinel-2" in x:
        return np.asarray(array),src
    else: #"Spot-P"
        return np.squeeze(np.asarray(array),axis=-1),src

def get_buf(patch_size):
    if (patch_size % 2) == 0:
        buf = patch_size-patch_size//2
    else:
        buf = patch_size-patch_size//2-1
    return buf

def get_batch(sensor_array,patch_size,sensor_src,r_,c_):
    r, c =  latlong2pixels(sensor_src, r_, c_)# pixel location corresponding to s1_vv(r,c) 

    if (patch_size % 2) == 0:
        buf = patch_size-patch_size//2
        patch = sensor_array[:,r-buf:r+buf,c-buf:c+buf]
    else:
        buf = patch_size-patch_size//2-1
        patch = sensor_array[:,r-buf:r+buf+1,c-buf:c+buf+1 ]
    
    if patch_size ==1:
        patch = np.squeeze(patch)

    # replace buf par patch
    return patch

S1, S1_src = read_sensor(S1_list_path)
S2, S2_src = read_sensor(S2_list_path)
SpotMs, SpotMs_src = read_sensor(SpotMs_list_path)
SpotP, SpotP_src = read_sensor(SpotP_list_path)

# extract patches a,b that correspond to the same area in A/B
ps_buf_SpotP = 32 # patch size of s1 vvv
ps_SpotMs = 8 # patch size of s1 vvv
ps_S1 = 9 # patch size of s1 vvv
ps_S2 =1

count = 0
nlat, nlong = get_image_latlong(S1_list_path[0])

t = time.time()

# we do the loop over the s1 file
# for r in tqdm(range(ps_S1, S1.shape[1]-ps_S1)):
#     for c in range(ps_S1, S1.shape[2]-ps_S1):
for i in range(0,(S1.shape[1]-2*ps_S1)*(S1.shape[2]-2*ps_S1)):
    c = ps_S1 + i//((S1.shape[1] - 2*ps_S1)) 
    r = ps_S1 + i - (i//(S1.shape[1] - 2*ps_S1))*(S1.shape[1]-2*ps_S1)
    try:
        patch_s1 = torch.as_tensor(get_batch(S1,ps_S1,S1_src,nlat[r,c],nlong[r,c]).astype('float32'))
        patch_s2 = torch.as_tensor(get_batch(S2,ps_S2,S2_src,nlat[r,c],nlong[r,c]).astype('float32'))
        patch_SpotMs = torch.as_tensor(get_batch(SpotMs,ps_SpotMs,SpotMs_src,nlat[r,c],nlong[r,c]).astype('float32'))
        patch_SpotP = torch.as_tensor(get_batch(SpotP,ps_buf_SpotP,SpotP_src,nlat[r,c],nlong[r,c]).astype('float32'))
    except IndexError:
        breakpoint()

    # dicts = {}
    # # dicts["PAN"] = torch.as_tensor(patch_spot_p)
    # # dicts["MS"] = torch.as_tensor(patch_spot_ms)
    # # dicts["S1"] =
    # # dicts["S2"] =
    
    # print(f'shape is {patch_spot_ms.shape[1]}')
    count+= 1
    print(f"i:{i} r:{r} c:{c}")
elapsed = time.time() - t
print(elapsed)
# except:
#     print('we have a problem')
#     breakpoint()
breakpoint()