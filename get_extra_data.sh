#!/bin/bash

cd data

wget http://www.z-yt.net/tmp/densecap/image2regions.json.gz
echo Unzip ...
gunzip image2regions.json.gz

wget http://www.z-yt.net/tmp/densecap/regions_dict.json.gz
echo Unzip ...
gunzip regions_dict.json.gz

