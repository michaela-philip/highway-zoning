# highway-zoning

This repository contains code for my paper on zoning and highway placement. 

## Data Sources
Full-count census data are provided by IPUMS/NBER
Highway shapefiles are graciously provided by Professors Taylor Jaworski and Carl Kitchens
Zoning data come from a variety of sources, but shapefiles are created in ArcGIS Pro
Information on street names comes from Steve Morse

## Code Structure
Code should be run in the following order:
1. import_census.py
2. clean.py
3. create_sample.py
These three files are written to be scalable - each script contains a section at the bottom to be updated with addition of new cities. Code outside of these sections should not be changed. 

## Scraping Scripts
Due to variation in sources for historical street names, scrape_streets.py contains unique functions for each city's name changes. New functions to incorporate an additional city's name changes should be written in scrape_streets.py and called in clean.py

## NOTE
Folder cnn is a modified clone of Michael Pollman's repository, found [here](https://github.com/michaelpollmann/spatialTreat-example.git).