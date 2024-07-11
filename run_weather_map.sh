#!/bin/bash
#source /opt/conda/mini/etc/profile.d/conda.sh
source ~/.bashrc
conda activate data-vis
python3 /home/sjohn/projects/data-visuals/weather_map/weather_map.py 2>&1 | tee /home/sjohn/projects/data-visuals/weather_map/error_log.txt
python3 /home/sjohn/projects/data-visuals/weather_map/video_generator.py


