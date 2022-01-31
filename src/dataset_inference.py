import rasterio
import numpy as np
from tqdm import tqdm
from typing import List
import time
from torch.utils.data import Dataset as BaseDataset
import torch
from pathlib import Path

class Dataset_inference(BaseDataset):
    def __init__(
            self,
            root: str,
            dataset: str,
            sensor: list, #todo
            ref = "S1"
    ):
        self.dataset = dataset
        self.ps_buf_SpotP = 32 # patch size Spot
        self.ps_SpotMs = 8 # patch size of MS  
        self.ps_S1 = 9 # patch size of s1
        self.ps_S2 = 1
        self.ps = {"S1":9,"S2":1,"PAN":32,"MS":8}
        self.sensor=sensor
        self.ref = ref
        
        for x in sensor:
            if x == "S1":
                S1_list_path = [f"{root}/{dataset}/Sentinel-1/{x}_ASC_CONCAT_S1.tif" for x in ["VV","VH"]]
                self.S1, self.S1_src = self.__read_sensor(S1_list_path)

                if self.ref == "S1":
                    self.refpath = S1_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath) # ref 
                    self.n_row = self.S1.shape[1]
                    self.n_col = self.S1.shape[2]

            elif x == "S2":
                S2_list_path =  [f"{root}/{dataset}/Sentinel-2/{x}_REUNION_2017_CONCAT_S2_GAPF.tif" for x in ["B2","B3","B4","B8","NDVI","NDWI"]]
                self.S2, self.S2_src = self.__read_sensor(S2_list_path)

                if self.ref == "S2":
                    self.refpath = S2_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath) # ref 
                    self.n_row = self.S2.shape[1]
                    self.n_col = self.S2.shape[2]

            elif x == "Spot":
                SpotMs_list_path = [f"{root}/{dataset}/Spot-MS/2017_mosa_S6_MS_TOA_ortho_kalideos_v4.tif"]
                self.SpotMs, self.SpotMs_src = self.__read_sensor(SpotMs_list_path)
                SpotP_list_path = [f"{root}/{dataset}/Spot-P/2017_mosa_S6_P_TOA_ortho_kalideos_v4.tif"]
                self.SpotP, self.SpotP_src = self.__read_sensor(SpotP_list_path)

                if self.ref == "Spot": # SpotP by default
                    self.refpath = SpotP_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath) # ref 
                    self.n_row = self.SpotP.shape[1]
                    self.n_col = self.SpotP.shape[2]  

    def get_source(self):
        if self.ref == "S1":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "S2":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "Spot":
            self.src = rasterio.open(self.refpath)
        return self.src

    def __get_image_latlong(self, raster_path: str)-> (List[float], List[float]):
        with rasterio.open(raster_path) as src:
            band1 = src.read(1)
            print('Reg image has shape', band1.shape)
            height = band1.shape[0]
            width = band1.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            lons= np.array(xs)
            lats = np.array(ys)
            return lats, lons

    def __latlong2pixels(self, raster_src, lat, lon):
        py, px = raster_src.index(lon, lat)
        #print(f"px:{px} py:{py}")
        return py, px

    def __read_sensor(self, input: List):
        array = []
        for i,x in enumerate(input):
            with rasterio.open(input[i]) as src:
                print(f"reading.. {Path(x).stem}")

                ts_array = src.read()
                if "Spot-MS" in x: # on peut le faire plus tard pour toutes les "array" et éviter ce if
                    for c_ in range(0,ts_array.shape[0]):
                        ts_array[c_] = (ts_array[c_] - np.min(ts_array[c_]))/(np.max(ts_array[c_]) - np.min(ts_array[c_]))
                else:
                    ts_array = (ts_array - np.min(ts_array))/(np.max(ts_array) - np.min(ts_array))

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

    def __get_batch(self, sensor_array,patch_size,sensor_src,r_,c_):
        r, c =  self.__latlong2pixels(sensor_src, r_, c_)# pixel location corresponding to s1_vv(r,c) 
        if (patch_size % 2) == 0:
            buf = patch_size-patch_size//2
            patch = sensor_array[:,r-buf:r+buf,c-buf:c+buf]
            #print(f"buf:{buf} r:{r} c:{c} patch shape:{patch.shape}")
        else:
            buf = patch_size-patch_size//2-1
            patch = sensor_array[:,r-buf:r+buf+1,c-buf:c+buf+1]
            #print(f"buf:{buf} r:{r} c:{c} patch shape:{patch.shape}")
        if patch_size == 1:
            patch = np.squeeze(patch)
        # replace buf par patch
        return patch

    def __len__(self):
        self.size = (self.n_row -2*self.ps[self.ref])*(self.n_col -2*self.ps[self.ref])
        return self.size

    def numtarget(self):
        if self.dataset in ["data_reunion","data_ToyReunion"]:
            self.num_target = 11
        else:
            self.num_target = 7
        return self.num_target

    def shape(self):
        return self.n_row, self.n_col


    def __getitem__(self, i):
            #t o check

        c = self.ps[self.ref] + i//((self.n_row - 2*self.ps[self.ref])) 
        r = self.ps[self.ref] + i - (i//(self.n_row - 2*self.ps[self.ref]))*(self.n_row-2*self.ps[self.ref])
        dicts = {}
        dicts["ind"] = {}
        dicts["ind"]["lat"] = self.nlat[r,c]
        dicts["ind"]["long"] = self.nlong[r,c]
        dicts["ind"]["r"] = r
        dicts["ind"]["c"] = c

        for x in self.sensor:
            if x == "S1":
                dicts[x] = torch.as_tensor(self.__get_batch(self.S1,self.ps["S1"],self.S1_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
            elif x == "S2":
                dicts[x] = torch.as_tensor(self.__get_batch(self.S2,self.ps["S2"],self.S2_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
            elif x == "Spot":
                dicts["PAN"] = torch.as_tensor(self.__get_batch(self.SpotP,self.ps["PAN"],self.SpotP_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
                dicts["MS"] = torch.as_tensor(self.__get_batch(self.SpotMs,self.ps["MS"],self.SpotMs_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
        return dicts