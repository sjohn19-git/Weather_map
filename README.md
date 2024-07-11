# Data Visuals Repository
This repository is where all code created for the [Data Visuals Project](https://app.asana.com/0/1205732312106349/1205732312106361) will be stored.  

*README updated :: 20 October, 2023 by Gabe Paris*

### Author
Sebin John

## Command Line Usage
python ./weather_map.py st et 

st: start time. This should be of the format "(yyyy,mm,dd,hh)" and in UTC time

et: end time. This should be of the format "(yyyy,mm,dd,hh)" and in UTC time

## Description

weather_map.py creates a weather map of Alaska at the specified start time and end time.
This code uses data from the AEC database.
This includes temperature, wind magnitude, wind direction, and pressure.
The plotting style is kept similar to meteorological standards.

map output: ./maps/{datetime}.png
If the file size of this folder exceeds 500 MB, the oldest map will be deleted each time this code is run. 

## Required Files

For this code to run there should be these files in the same directory as the code.

station list file - contains comma-delimited station names to be plotted

grids directory - contains topography grid and shading grid for the map

cpts directory - Specifies color map to be used. (This is hardcoded)

Alaska_network_station_location.csv - Locations of the stations in the AK network

## Debugging

./error_log.txt file stores the standard output and errors for debugging


## Dependencies

weather_map.py depends on the following:  

1. Conda python  
2. Matplotlib  
3. Cartopy
4. tqdm
5. numpy
6. obspy
7. cartopy
8. metpy
9. dill
10. argparse
11. ast
12. pytz
## Caveats

This code will only work in the Antelope installed systems connected to the AEC database

## Example

python ./weather_map.py "(2024,23,00)" "(2024,23,01)"

```
> 100%|██████████| 69/69 [07:01<00:00,  6.11s/it]
> 100%|██████████| 63/63 [03:35<00:00,  3.42s/it]
> Figure size in centimeters: 25.40 x 20.32 cm
```

* Please expect a couple of status bars. We are aware of the long run time and intend to fix it in the future.
