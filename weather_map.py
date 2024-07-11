#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 10 10:04:03 2023

@author: sebin john
email:sjohn19@alaska.edu
"""
#-------------------------------------------------------------
"""
This code generates a weather map using data in the AEC database
This includes temperature, wind, wind direction, and pressure
The plotting style is kept similar to meteorological standards.
It reads files from ./grids and ./cpts directory for plotting. 
If these folders are not available code will fail
Stations to be plotted are determined by the ./station_list.txt file
Alaska_network_station_location.csv should be in the script directory
map output: ./maps/{datetime}.png
"""

import os
from tqdm import tqdm
from os.path import join as jn
import sys
# Get the directory of the script
if getattr(sys, 'frozen', False):  # if the script is frozen (e.g., by PyInstaller)
    script_dir = os.path.dirname(sys.executable)
elif '__file__' in globals():  # if __file__ is defined (i.e., not in interactive mode)
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:  # if __file__ is not defined (e.g., in interactive mode or during import)
    script_dir = os.getcwd()
root_path=script_dir
os.chdir(root_path)
'''following 5 lines of code ensures that we are loading
    updated version of wf2obspy from aec antelope python'''
import importlib.util
module_path = '/usr/local/aec/'+os.environ['ANT_VER']+'/data/python/wf2obspy.py'
spec = importlib.util.spec_from_file_location('wf2obspy', module_path)
wf2obspy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf2obspy)
import numpy as np
import numpy.ma as ma
from obspy import UTCDateTime
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.units import units
from metpy.plots import StationPlot
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import dill
#import inspect
import argparse
import ast
from datetime import datetime
import pytz
#source_file = inspect.getsourcefile(wf2obspy)
#print("Source file location:", source_file)



def get_folder_size(folder_path):
    ''' This function returns the size of a folder
        
        :param folder_path: (string) Path to the folder
    '''
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    # Convert bytes to megabytes
    total_size_mb = total_size / (1024 * 1024)
    return total_size_mb


def valid_ind(arr): 
    '''This function identifies data gap 
        and returns a boolean array according to gaps
        
        :param arr: (list/array) array or list of the data'''
    
    m=[]
    for ele in arr:
        try:
            if np.isnan(ele.magnitude):
                m.append(False)
            else:
                m.append(True)
        except:
            m.append(False)
    return m


def haversine(lat1, lon1, lat2, lon2): 
    ''' This is a function to calculate the distance between two points
        Returns distance in m
        
        :param lat1: (int) latitude of first point
        :param lon1: (int) longitude of first point
        :param lat2: (int) latitude of second point
        :param lon2: (int) longitude of second point
        '''
        
    #Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Radius of the Earth in kilometers (mean value)
    radius = 6371.0
    # Calculate distance
    distance = radius * c
    return distance


def filt_dist(dis):
    '''This function filters out only one station in a 
        circular radius specified by dis
        If dis is zero then no stations are excluded
      This function also finds stations where both wind 
      and pressure data are not nan and filter out only
      data corresponding to such stations.
      This same criterion is used to filter temperature data
      Returns station_id,pressure,uf,vf,final_lon,final_lat,temp
      station_id: list of selected stations
      pressure: (list/array) pressure data
      uf: (list/array) u component of wind
      vf: (list/array) v component of wind
      final_lon: (list/array) longitude of stations
      final_lat: (list/array) latitude of stations
      temp: (list/array) temperature data
      This function makes use of ./saved_variables.pkl
      The function assumes that this is present in the working directory
      ./saved_variables.pkl is created by download_data(param) function
      
      :param dis: (int) radius distance '''
     
    # saved_variables.pkl file is the processed data created using download function
    with open('saved_variables.pkl', 'rb') as file: 
        loaded_variables = dill.load(file)
    station_id,pressure,uf,vf,final_lon,final_lat,temp = loaded_variables
    val_p=valid_ind(pressure)
    val_w=valid_ind(uf)
    sele1=np.logical_or(val_p, val_w)
    final_lat=final_lat[sele1]
    final_lon=final_lon[sele1]
    uf=uf[sele1]
    vf=vf[sele1]
    pressure=pressure[sele1]
    min_distance = dis
    sele = []
    s_ids = set()
    for i in range(len(final_lon)):
        if i not in s_ids:
            sele.append(i)  # Add the current station to the selected list
            s_ids.add(i)
    
            for j in range(i + 1, len(final_lon)):
                if j not in s_ids:
                    distance = haversine(final_lat[i], final_lon[i], final_lat[j], final_lon[j])
                    if distance <= min_distance:
                        # Skip this station as it is too close to the already selected one
                        s_ids.add(j)
    final_lat=final_lat[sele]
    final_lon=final_lon[sele]
    uf=uf[sele]
    vf=vf[sele]
    pressure=pressure[sele]
    station_id=station_id[sele]
    temp=np.array(temp)[sele]
    return station_id,pressure,uf,vf,final_lon,final_lat,temp

def download_data(starttime,endtime): 
    ''' This function downloads temp, pressure, and wind data
        compute the mean value of the specified time range.
        handle station availability
        returns station_id,pressure,uf,vf,final_lon,final_lat,temp
        station_id: list of selected stations
        pressure: (list/array) pressure data
        uf: (list/array) u component of wind
        vf: (list/array) v component of wind
        final_lon: (list/array) longitude of stations
        final_lat: (list/array) latitude of stations
        temp: (list/array) temperature data
        input files of this function are 
        (assumes to be in the current directory)
        ./Alaska_network_station_location.csv: station locations
        ./station_list.txt: list of station files
        output file of this function is
        (saves in the current directory)
        ./saved_variables.pkl: contain all parameters function returns

        :param starttime: (datetime/UTCDateTime) start time
        :param endtime: (datetime/UTCDateTime) end time
        '''
        
    stations = []
    lonw,lonpr = np.array([]),np.array([])
    latw,latpr = np.array([]),np.array([])
    # loading file containing station location
    stat_loc = pd.read_csv("./Alaska_network_station_location.csv")
    with open("./station_list.txt", 'r') as f:  # loading stations to plot temperature map
        content = f.read()
        sta = content.replace('\n', '').split(',')
        stations.extend(sta)
    pr,u,v = [],[],[]
    for jj in tqdm(range(len(stations))):
        sta=stations[jj]
        if endtime.hour==0:#handling day boundary
            endtime=endtime-1
        try:
            st = wf2obspy.get_waveforms("AK", sta,"EP", "LWD", starttime, endtime, dbname="/aec/db/weather/weather")  # getting data
            ma.set_fill_value(st[0].data,np.nan)
            wds=st[0].data.data*units.deg
            st = wf2obspy.get_waveforms("AK", sta, "EP", "LWS", starttime, endtime, dbname="/aec/db/weather/weather")
            ma.set_fill_value(st[0].data,np.nan)
            wss=(st[0].data.data)*0.1*units('m/s')
            us,vs=mpcalc.wind_components(wss.to('knots'), wds)
            ui=np.nanmean(us) # calculating mean wind speed in u direction
            vi=np.nanmean(vs)
            if ui==np.nan:
                break
            u.extend([ui])
            v.extend([vi])
            lonw = np.append(
                lonw, (stat_loc[stat_loc["Station Code"] == sta]["Longitude"].iloc[0]))
            latw = np.append(
                latw, (stat_loc[stat_loc["Station Code"] == sta]["Latitude"].iloc[0]))
        except:
            u.extend([np.nan])
            v.extend([np.nan])
            latw = np.append(latw, 0)
            lonw = np.append(lonw, 0)
        try: #downloading pressure
            st = wf2obspy.get_waveforms(
                "AK", sta, "EP", "BDO", starttime, endtime)
            ma.set_fill_value(st[0].data,np.nan)
            pr.extend([np.nanmean(st[0])]) 
            lonpr = np.append(
                lonpr, (stat_loc[stat_loc["Station Code"] == sta]["Longitude"].iloc[0]))
            latpr = np.append(
                latpr, (stat_loc[stat_loc["Station Code"] == sta]["Latitude"].iloc[0]))
        except:  # handling no data availability
            pr.extend([np.nan])
            latpr = np.append(latpr, 0)
            lonpr = np.append(lonpr, 0)
    lonpr[np.where(lonpr[lonpr < 0])] = lonpr[np.where(lonpr[lonpr < 0])]+360
    lonw[np.where(lonw[lonw < 0])] = lonw[np.where(lonw[lonw < 0])]+360 #modifing the lon to 0-360 range
    pr,u,v = np.array(pr),np.array(u,dtype=object),np.array(v,dtype=object)
    #removing unavailable stations
    nans = np.logical_or(lonw==0, lonpr == 0) 
    final_lon = lonpr[~nans]
    final_lat = latpr[~nans]
    station_id = np.array(stations)[~nans]
    pressure=pr[~nans]*units.Pa
    uf=u[~nans]
    vf=v[~nans]
    temp=[]
    #downloading temperature data
    for ll in tqdm(range(len(station_id))):
        sta=station_id[ll]
        try:
            st=wf2obspy.get_waveforms("AK", str(sta), "EP", "LKO", starttime, endtime,dbname="/aec/db/weather/weather")#getting data
            temp.extend([np.nanmean(st[0])/10])
        except:
            temp.extend([np.nan])
    with open('saved_variables.pkl', 'wb') as file:
        dill.dump([station_id,pressure,uf,vf,final_lon,final_lat,temp], file)
    return station_id,pressure,uf,vf,final_lon,final_lat,temp

def plot(station_id,pressure,uf,vf,final_lon,final_lat,
        name,size,temp):
    ''' This function plots the final map and saves in the 
       ./maps directory. This function also makes sure
       that the size of the directory does not exceeding 500 MB
        
        :param station_id: list of station names
        :param pressure: (list/array) pressure data 
        :parma uf: (list/array) u component of wind
        :param vf: (list/array) v component of wind
        :param final_lon: (list/array) longitudes of stations
        :param final_lat: (list/array) latitudes of stations
        :param name: (datetime/UTCDateTime) time of the map generation for title
        :param size: (int) size of the symbols in the map
        :param temp: (list/array) temperature data
        '''
        
    #plotting function
    legws=np.arange(0,60,10)*units('knots')
    legws[3]=25*units('knots') # manually defining labels
    legws[4]=50*units('knots')
    legws[3]=25*units('knots')
    legws[5]=75*units('knots')
    legws = np.insert(legws, 1,5*units('knots') )
    legwd=np.zeros(len(legws))*units('degree')
    legus,legvs=mpcalc.wind_components(legws, legwd)
    fig, mx = plt.subplots(nrows=1,ncols=1, figsize=(10,8),dpi=300)
    bounds_ax1 = [0, -0.05, 0.65, 0.65]
    bounds_ax2 = [0.405, 0.19, 0.2, 0.33]
    mx.set_visible(False)
    ax1=fig.add_axes(bounds_ax1,projection=ccrs.AlbersEqualArea(central_longitude=-154, central_latitude=64, standard_parallels=(55, 65)))
    #ax1=fig.add_axes(bounds_ax1,projection=request.crs)
    ax1.add_feature(cfeature.OCEAN,facecolor='aliceblue')
    ax1.add_feature(cfeature.LAND,facecolor='ivory')
    ax1.add_feature(cfeature.COASTLINE, lw=0.2)
    # Add coastlines and borders
    # ax.coastlines(resolution='50m', linewidth=0.5)
    #ax1.add_image(request, 4)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.set_extent([-165,230,55, 72])  # Set the extent to cover Alaska
    stationplot = StationPlot(ax1, final_lon, final_lat, clip_on=True,
                              transform=ccrs.PlateCarree(), fontsize=size)
    stationplot.plot_parameter('NE', pressure, plot_units=units.hPa,
                                formatter=lambda v: format(v/10, '.0f'), color="k")
    uplo,vplo=[],[]
    for i in range(len(uf)):
        try:
            uplo.append(uf[i].magnitude)
            vplo.append(vf[i].magnitude)
        except:
            uplo.append((np.nan))
            vplo.append(np.nan)
    
    stationplot.plot_barb(uplo, vplo)
    ax1.text(200,72.6,"last updated: "+name,transform=ccrs.PlateCarree())
    scatter = ax1.scatter(final_lon, final_lat, marker='o', transform=ccrs.PlateCarree(),s=35,c=temp,cmap="jet",vmin=-50,vmax=30)
    
    ax2 = fig.add_axes(bounds_ax2)
    ax2.set_xlim([0,10])
    ax2.set_ylim([0,10])
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    stationplot = StationPlot(ax2, [1,4.25,7.25,1,4.25,7.25], [8.3,7.8,7.8,6.3,6.3,6.3],clip_on=True,fontsize=size)
    stationplot.plot_barb(legus[:6], legvs[:6])
    ax2.text(2.2,9.4,"Wind speed (knot,kt)",fontsize=size+1)
    ax2.text(5,8.2,"5 kt",fontsize=size)
    ax2.text(1.8,8.2,"< 3 kt",fontsize=size)
    ax2.text(8,8.2,"10 kt",fontsize=size)
    ax2.text(2,6.8,"20 kt",fontsize=size)
    ax2.text(5,6.8,"25 kt",fontsize=size)
    ax2.text(8,6.8,"50 kt",fontsize=size)
    ax2.text(1.4,5.4,"1055  Pressure (hPa)",fontsize=size+1,color="k")
    ax2.scatter(1.7,4.6,s=35,cmap="jet",vmin=-50,vmax=30,c=[-10])
    ax2.text(2.5,4.4,"Temperature",fontsize=size+1)
    ax2.set_facecolor("ivory")
    ax2.text(2.5,3.4,"Fahrenheit (\u00B0F)",fontsize=size)
    cax_position = [0.42, 0.24, 0.17, 0.02]  # [left, bottom, width, height]
    cax = fig.add_axes(cax_position)
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal', label='Celcius (\u00B0C)')
    cbar.ax.tick_params(labelsize=size)
    cbar.ax.xaxis.label.set_size(size)  
    tick_positions = [-50, -30,-10,10, 30]  # specify your desired tick positions
    cbar.set_ticks(tick_positions)
    ax4 = cax.secondary_xaxis('top')
    ax4.tick_params(labelsize=size)
    # Function to convert Celsius to Fahrenheit
    def celsius_to_fahrenheit(celsius):
        celsius=np.array(celsius)
        return celsius * 9/5 + 32
    fahrenheit_labels = [str(celsius_to_fahrenheit(temp))[:-2] for temp in tick_positions]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(fahrenheit_labels)
    
    fig.savefig(root_path+'/maps/'+name+".png",bbox_inches='tight', pad_inches=0.1)
    fig_width_cm = fig.get_size_inches()[0] * 2.54
    fig_height_cm = fig.get_size_inches()[1] * 2.54
    print("Figure size in centimeters: {:.2f} x {:.2f} cm".format(fig_width_cm, fig_height_cm))
    if get_folder_size(jn(root_path,"maps"))>500:
        open_g=[]
        for root,dire,file in os.walk(jn(root_path,"maps")):
            for name in file:
                open_g.append(name)
        open_g.sort()
        os.remove((os.path.join(root,open_g[0])))

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="weather plot")
        # Add command-line arguments for st and et
        parser.add_argument("st", type=str, help="Start time (yyyy,mm,dd,hh)")
        parser.add_argument("et", type=str, help="End time (yyyy,mm,dd,hh)")
        # Parse the command-line arguments
        args = parser.parse_args()
        stt = ast.literal_eval(args.st)
        ett = ast.literal_eval(args.et)
        st = UTCDateTime(stt[0], stt[1], stt[2], stt[3])
        et = UTCDateTime(ett[0], ett[1], ett[2], ett[3])
    except:
      # Manually define start and end  if arguments not provided
          print("using default st and et")
          timec = datetime.utcnow().replace(microsecond=0, second=0, minute=0)
          time = UTCDateTime(timec) - (6 * 3600)
          st = time  # Default start time
          et = st + 3600  # Default end time
    tit=st+(et-st)/2
    akdt_timezone = pytz.timezone("America/Anchorage")
    tit_akdt = tit.datetime.replace(tzinfo=pytz.UTC).astimezone(akdt_timezone)
    name=tit_akdt.strftime("%Y-%m-%d %H:%M")
    station_id,pressure,uf,vf,final_lon,final_lat,temp = download_data(st,et)
    station_id,pressure,uf,vf,final_lon,final_lat,temp=filt_dist(0)
    plot(station_id,pressure,uf,vf,final_lon,final_lat,name,8,temp)
    

        


