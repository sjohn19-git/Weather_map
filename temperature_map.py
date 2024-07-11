#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:33:06 2023

@author: sjohn
"""

import wf2obspy
import numpy as np
from obspy import UTCDateTime
import pygmt
import pandas as pd
# import xarray as xr
# from scipy.interpolate import interp2d
# from scipy.interpolate import griddata
# import geopandas as gpd



def download_data():
    starttime=UTCDateTime.now()-48*3600 #defining start time
    endtime=UTCDateTime.now()-47*3600 #defining end time
    stations=[]
    lon=np.array([])
    lat=np.array([])
    stat_loc=pd.read_csv("./Alaska_network_station_location.csv") #loading file containing station location
    with open("./station_list",'r') as f: #loading stations to plot temperature map
        content = f.read()
        sta=content.replace('\n','').split(',')
        stations.extend(sta)
    data=[]
    for sta in stations:
        try:
            st=wf2obspy.get_waveforms("AK", sta, "EP", "LKO", starttime, endtime,dbname="/aec/db/weather/weather")#getting data
            data.extend([np.nanmean(st[0])]) #taking mean temperature
            lon=np.append(lon,(stat_loc[stat_loc["Station Code"]==sta]["Longitude"].iloc[0]))# extracting station location
            lat=np.append(lat,(stat_loc[stat_loc["Station Code"]==sta]["Latitude"].iloc[0]))
        except: # handling no data availability
            data.extend([1500])
            lat=np.append(lat,0)
            lon=np.append(lon,0)
    lon[np.where(lon[lon<0])]=lon[np.where(lon[lon<0])]+360
    data=np.array(data)
    data=data/10 # converting to celcius
    lon=np.where(data == 1500, 0, lon)
    nan_indices = np.isnan(data)
    lon[nan_indices] = 0.0
    data=np.where(data == 1500, np.nan, data)
    return stations,data,lat,lon

# def interpolate(lon,lat,data): #function to interpolate temp in space
#     lats=np.arange(50,69,0.01)
#     lons=np.arange(180,230,0.01)
#     xi,yi=np.meshgrid(lons,lats)
#     nan_indices = np.isnan(data)
#     interp_func = interp2d(lon[~nan_indices], lat[~nan_indices], data[~nan_indices], kind='nearest',fill_value=np.nan)
#     grid=interpolated_value = interp_func(lons, lats)
#     grid[grid>=np.nanmax(data)]=np.nan
#     grid[grid<=np.nanmin(data)]=np.nan
#     grid=griddata((lon[~nan_indices], lat[~nan_indices]), data[~nan_indices], (xi, yi), method='nearest',fill_value=1)
#     temp_grid = xr.DataArray(grid, dims=["lat", "lon"], coords={"lat": lats, "lon": lons})
#     return temp_grid


def plot_map(stations,lat,lon,data): #function to plot map
    title=(UTCDateTime.now()-0.5*3600).strftime("%Y-%m-%d %H:%M")
    #grid1=pygmt.datasets.load_earth_relief(resolution='02m', region=[150,260, 30, 78])
    #grid1.to_netcdf("dem.nc")
    #cpt=pygmt.makecpt(cmap="./cpts/tem.cpt",series=[-33,33,0.1],output="temperature.cpt",reverse=False)
    #shade=pygmt.grdgradient(grid="./grids/dem.nc",radiance="p")
    proj="L-159/35/33/85/20c"
    fig=pygmt.Figure()
    reg="178/45/260/69r"
    with pygmt.config(MAP_FRAME_TYPE="plain"):
        fig.basemap(region=reg, projection=proj,frame="f")
    fig.grdimage(grid="./grids/dem.nc",region=reg,projection=proj, cmap="./cpts/bath1.cpt",nan_transparent=True,shading='+a300+nt0.8')
    fig.coast(region=reg, projection=proj,shorelines=True,borders=["1/0.5p,grey"],area_thresh='600',dcw=["RU+g211/211/211@50","CA+g211/211/211@50"])
    fig.coast(region=reg, projection=proj,borders=["1/0.5p,black"],area_thresh='600',dcw=["US.AK"])
    fig.plot(x=lon,y=lat,style="c0.5c",color=data,cmap="./cpts/temperature.cpt",pen="black", projection=proj)
    fig.text(text=str(title)[0:16],x=201,y=75.5,projection=proj,font="13p,Helvetica-Bold")
    #fig.grdimage(grid=temp_grid,region=reg,projection=proj, cmap="./cpts/temperature.cpt",nan_transparent=True)
    fig.colorbar(projection=proj, cmap="./cpts/temperature.cpt",frame=["x+lTemperature", "y+lC"],position="n0.57/0.07+w8c/0.5c+h")
    fig.show()
    fig.savefig("/home/sjohn/Data_visual/fig.png")


stations,data,lat,lon=download_data()
# temp_grid=interpolate(lon,lat,data)
plot_map(stations,lat,lon,data)



# starttime=UTCDateTime(2022,1,1)
# endtime=UTCDateTime(2022,1,1,10)

# # cpt=pygmt.makecpt(cmap="./temperature.cpt",series=[-500,1000,1000],output="temp.cpt",reverse=False)
# for sta in stations:
#     wf2obspy.get_waveforms("AK", sta, "*", "BHZ", starttime, endtime)
