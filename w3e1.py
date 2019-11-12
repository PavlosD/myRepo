# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 07:00:07 2019

@author: Paul
"""
import geopandas


countries = geopandas.read_file("zip:world_m.zip")
cities=geopandas.read_file("zip:cities.zip")



countries.plot()



cities.plot(marker='*', color='green', markersize=5);

cities = cities.to_crs(countries.crs)

base = countries.plot(color='white', edgecolor='black')

cities.plot(ax=base, marker='o', color='red', markersize=5);