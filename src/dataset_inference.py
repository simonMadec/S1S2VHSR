import rasterio
import numpy as np
from tqdm import tqdm
from typing import List
import time
from torch.utils.data import Dataset as BaseDataset
import torch
from pathlib import Path
from glob import glob
from rasterio.windows import from_bounds
from rasterio.transform import Affine as A

class Dataset_inference(BaseDataset):
    def __init__(
            self,
            root: str,
            dataset: str,
            bbox,
            sensor: list, #todo
            ref = "S1"
    ):
        self.dataset = dataset
        self.ps_buf_SpotP = 32 # patch size Spot
        self.ps_SpotMs = 8 # patch size of MS  
        self.ps_S1 = 9 # patch size of s1
        self.ps_S2 = 1
        self.ps = {"S1":9,"S2":1,"Spot":32,"PAN":32,"MS":8}
        self.sensor=sensor
        self.ref = ref
        self.norm = np.load(Path(root) / dataset / "norm_info.npy" , allow_pickle=True).item()
        print(self.ref)

        for x in sensor:
            assert x in ["S1","S2","Spot"],"Wrong sensor name"

            if x == "S1":
                S1_list_path = [glob(f"{root}/{dataset}/Sentinel-1/{x}_ASC_CONCAT_S1.tif")[0] for x in ["VV","VH"]]
                # S1_list_path = [f"{root}/{dataset}/Sentinel-1/{x}_ASC_CONCAT_S1.tif" for x in ["VV","VH"]]
                self.S1, self.S1_src = self.__read_sensor(S1_list_path,bbox)

                if self.ref == "S1":
                    self.refpath = S1_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath,bbox) # ref 
                    self.n_row = self.S1.shape[1]
                    self.n_col = self.S1.shape[2]

            elif x == "S2":
                S2_list_path =  [glob(f"{root}/{dataset}/Sentinel-2/{x}_*GAPF.tif")[0] for x in ["B2","B3","B4","B8","NDVI","NDWI"]]
                # S2_list_path =  [f"{root}/{dataset}/Sentinel-2/{x}_REUNION_2017_CONCAT_S2_GAPF.tif" for x in ["B2","B3","B4","B8","NDVI","NDWI"]]
                self.S2, self.S2_src = self.__read_sensor(S2_list_path,bbox)

                if self.ref == "S2":
                    self.refpath = S2_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath,bbox) # ref 
                    self.n_row = self.S2.shape[1]
                    self.n_col = self.S2.shape[2]

            elif x == "Spot":
                SpotMs_list_path = [glob(f"{root}/{dataset}/Spot-MS/*.tif")[0]]
                self.SpotMs, self.SpotMs_src = self.__read_sensor(SpotMs_list_path,bbox)
                SpotP_list_path = [glob(f"{root}/{dataset}/Spot-P/*.tif")[0]]
                self.SpotP, self.SpotP_src = self.__read_sensor(SpotP_list_path,bbox)

                if self.ref == "Spot": # SpotP by default
                    self.refpath = SpotP_list_path[0]
                    self.nlat, self.nlong = self.__get_image_latlong(self.refpath,bbox) # ref 
                    self.n_row = self.SpotP.shape[1]
                    self.n_col = self.SpotP.shape[2]  

    def get_source(self, bb: List):
        assert self.ref in ["S1","S2","Spot"],"Wrong sensor name"

        if self.ref == "S1":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "S2":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "Spot":
            self.src = rasterio.open(self.refpath)
            
        if not bb:
            print(f"we take all images for")
            return self.src
        else:
            ts_array = self.src.read(window=from_bounds(bb[0], bb[1], bb[2], bb[3], transform=self.src.profile['transform']))
            nw_tr = self.src.profile['transform']
            nw_tr= A(nw_tr[0],nw_tr[1],bb[0],nw_tr[3],nw_tr[4],bb[3])
            meta_data_dict = {"width": ts_array.shape[2], "height": ts_array.shape[1], "transform": nw_tr}
            kwds = self.src.profile
            kwds.update(**meta_data_dict) #change affine
            profile = kwds
            profile["count"]= ts_array.shape[0] 
            with rasterio.open('temmp_example45423_trashtododebug.tif','w',**profile) as ds_out:
                ds_out.write(ts_array)
                return ds_out


    def get_profile(self, bb: List):
        assert self.ref in ["S1","S2","Spot"],"Wrong sensor name"

        if self.ref == "S1":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "S2":
            self.src = rasterio.open(self.refpath)
        elif self.ref == "Spot":
            self.src = rasterio.open(self.refpath)
            
        if not bb:
            print(f"we take all images for")
            return self.src.profile
        else:
            ts_array = self.src.read(window=from_bounds(bb[0], bb[1], bb[2], bb[3], transform=self.src.profile['transform']))
            nw_tr = self.src.profile['transform']
            nw_tr= A(nw_tr[0],nw_tr[1],bb[0],nw_tr[3],nw_tr[4],bb[3])
            meta_data_dict = {"width": ts_array.shape[2], "height": ts_array.shape[1], "transform": nw_tr}
            kwds = self.src.profile
            kwds.update(**meta_data_dict) #change affine
            profile = kwds
            profile["count"]= ts_array.shape[0] 

            return profile


    def __get_image_latlong(self, raster_path: str, bb: List):
        with rasterio.open(raster_path) as src:
            if not bb:
                band1 = src.read(1)
            else:
                win = from_bounds(bb[0], bb[1], bb[2], bb[3], transform=src.transform)
                band1 = src.read(1,window=win)

            # band1 = src.read(1)
            print('Ref image has shape', band1.shape)
            height = band1.shape[0]
            width = band1.shape[1]
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            print('get lat and long info ..')
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            lons= np.array(xs)
            lats = np.array(ys)
            return lats, lons

    def __latlong2pixels(self, raster_src, lat, lon):
        py, px = raster_src.index(lon, lat)
        #print(f"px:{px} py:{py}")
        return py, px

    def __read_sensor(self, input: List, bb: List):
        array = []
        for i,x in enumerate(input):
            with rasterio.open(input[i]) as src:
                print(Path(input[i]).stem)
                # print(f"reading.. {Path(x).stem}")

                if not bb:
                    ts_array = src.read()
                else:
                    win = from_bounds(bb[0], bb[1], bb[2], bb[3], transform=src.transform)
                    print(f"source is {src.crs}")
                    ts_array = src.read(window=win)

                min = self.norm[Path(input[i]).stem]["min"]
                max = self.norm[Path(input[i]).stem]["max"]
                print(f"do norm  {min} {max}")
                # min = np.min(ts_array)
                # max = np.max(ts_array)
                #ts_array = src.read()
                if "Sentinel-1" in x:
                    ts_array = -10*np.log(ts_array)

                    # if "VV" in Path(x).stem:
                    #     min=-68.4
                    #     max=55.27
                    # else:
                    #     min=-49.1
                    #     max=62.45

                # print("Min Max Standardisation")
                # print(input[i])
                # print(Path(input[i]).stem)
                # print(np.unique(ts_array))
                ts_array = np.moveaxis(ts_array,0,-1)
                ts_array = (ts_array - min)/(max - min)
                # ts_array = (ts_array - min)/(max - min)
            array.append(ts_array)
            ts_array=None

        if "Sentinel-1" in x:
           
           return np.dstack(array).transpose(-1,0,1), src
           #return np.moveaxis(np.dstack(array),-1,0),src
        elif "Spot-MS" in x:
            return array[0].transpose(-1,0,1), src
            #return np.moveaxis(np.squeeze(np.asarray(array),axis=0),-1,0),src
            
        elif "Sentinel-2" in x:
            return np.asarray(array),src
            # tt = np.stack(array,axis=-1)
            # breakpoint()
            # tt.reshape(tt.shape[0],tt.shape[1],np.prod(tt.shape[2:]))
            # return tt, src
            #return np.asarray(array),src   ##before it was just this
        else: #"Spot-P"
            try:
                return array[0].transpose(-1,0,1), src
                #return np.squeeze(np.asarray(array),axis=-1),src
            except ValueError:
                breakpoint()

    def __get_batch(self, sensor_array,patch_size,sensor_src,r_,c_):
        r, c =  self.__latlong2pixels(sensor_src, r_, c_)# pixel location corresponding to s1_vv(r,c) 
        if (patch_size % 2) == 0:
            buf = patch_size-patch_size//2
            patch = sensor_array[:,r-buf:r+buf,c-buf:c+buf]
            # print(f"buf:{buf} patch-size : {patch_size}  r:{r} c:{c} patch shape:{patch.shape}")
        else:
            buf = patch_size-patch_size//2-1
            patch = sensor_array[:,r-buf:r+buf+1,c-buf:c+buf+1]

        # data check give 0 value if no data #todo
        if len(sensor_array.shape) == 4: # 
            if (patch.shape[-2] != patch_size) or (patch.shape[-3] != patch_size):
                print("debug debug carefull we have no data")
                print(patch.shape)

        elif len(sensor_array.shape) == 3: # 
            if (patch.shape[-1] != patch_size) or (patch.shape[-2] != patch_size):

                print(f"patch size {patch_size}")
                print(f"patch.shape[-1] {patch.shape[-1]}")
                print(f"patch.shape[-2] {patch.shape[-2]}")
                print(f"buf {buf}")
                print(f"r : {r}")
                print(f"c : {c}")
                print("making 0 patch ! ")
                patch = np.zeros((sensor_array.shape[0],patch_size,patch_size))

        if patch_size == 1:
            patch = np.squeeze(patch)

        return patch

    def __len__(self):
        self.size = (self.n_row -2*self.ps[self.ref])*(self.n_col -2*self.ps[self.ref])
        return self.size

    def numtarget(self):
        if "reunion" in self.dataset:
            self.num_target = 11
            
        elif "Reunion" in self.dataset:
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
                PAN = self.__get_batch(self.SpotP,self.ps["PAN"],self.SpotP_src,self.nlat[r,c],self.nlong[r,c])
                dicts["PAN"] = torch.as_tensor(PAN.astype('float32'))
                MS = self.__get_batch(self.SpotMs,self.ps["MS"],self.SpotMs_src,self.nlat[r,c],self.nlong[r,c])
                dicts["MS"] = torch.as_tensor(MS.astype('float32'))
        return dicts
        # for x in self.sensor:
        #     if x == "S1":
        #         dicts[x] = torch.as_tensor(self.__get_batch(self.S1,self.ps["S1"],self.S1_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
        #     elif x == "S2":
        #         dicts[x] = torch.as_tensor(self.__get_batch(self.S2,self.ps["S2"],self.S2_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
        #     elif x == "Spot":
        #         dicts["PAN"] = torch.as_tensor(self.__get_batch(self.SpotP,self.ps["PAN"],self.SpotP_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
        #         dicts["MS"] = torch.as_tensor(self.__get_batch(self.SpotMs,self.ps["MS"],self.SpotMs_src,self.nlat[r,c],self.nlong[r,c]).astype('float32'))
        # return dicts