#!/usr/bin/python

import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import os
import zipfile
from shapely import geometry
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import shutil

import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from glob import glob



##Make Bounds file
column_names = 'line fid time day year latitude longitude radalt totmag resmag diurnal geology resmagCM4'.split(' ')
print column_names
mag_data = pd.read_csv('./data/iron_river_mag.xyz.gz',
                       delim_whitespace=True, names=column_names, usecols=['latitude', 'longitude', 'totmag'])
mag_data.head() # This shows the first 5 entries of the DF.

p1 = pyproj.Proj(proj='latlong', datum='NAD27')
# WGS84 Latlong
p2 = pyproj.Proj(proj='latlong', datum='WGS84')
# WGS84 UTM Zone 16
p3 = pyproj.Proj(proj='utm', zone=16, datum='WGS84')
mag_data['long_wgs84'], mag_data['lat_wgs84'] = pyproj.transform(p1, p2,
                                                                 mag_data.longitude.values,
                                                                 mag_data.latitude.values)
mag_data['E_utm'], mag_data['N_utm'] = pyproj.transform(p1, p3,
                                                        mag_data.longitude.values,
                                                        mag_data.latitude.values)

mag_data['geometry'] = [geometry.Point(x, y) for x, y in zip(mag_data['long_wgs84'], mag_data['lat_wgs84'])]
mag_data = gpd.GeoDataFrame(mag_data, geometry='geometry', crs="+init=epsg:4326")

mag_data.to_csv('./data/mag.csv')
mag_data.head(3)



multipoints = geometry.MultiPoint(mag_data['geometry'])
bounds = multipoints.envelope

gpd.GeoSeries(bounds).to_file('./data/area_of_study_bounds.gpkg', 'GPKG')

coords = np.vstack(bounds.boundary.coords.xy)
map_center = list(coords.mean(1))[::-1]

map_center



##Get overlapping Landsat tiles

bounds = gpd.read_file('./data/area_of_study_bounds.gpkg')

WRS_PATH = './data/wrs2_descending.zip'
LANDSAT_PATH = os.path.dirname(WRS_PATH)

shutil.unpack_archive(WRS_PATH, os.path.join(LANDSAT_PATH, 'wrs2'))
zip_ref = zipfile.ZipFile('./data/wrs2_descending.zip')
zip_ref.extractall('./data/wrs2/')
zip_ref.close()

wrs = gpd.GeoDataFrame.from_file('./data/wrs2/wrs2_descending.shp')

wrs.head()

wrs_intersection = wrs[wrs.intersects(bounds.geometry[0])]

paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values

b = (paths > 23) & (paths < 26)
paths = paths[b]
rows = rows[b]

for i, (path, row) in enumerate(zip(paths, rows)):
    print('Image', i+1, ' - path:', path, 'row:', row)


s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz', compression='gzip')

#More data in google scenes
# google_scenes = pd.read_csv('https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz', compression='gzip')

s3_scenes.head(3)


# Empty list to add the images
bulk_list = []

# Iterate through paths and rows
for path, row in zip(paths, rows):
    
    print('Path:',path, 'Row:', row)
    
    # Filter the Landsat Amazon S3 table for images matching path, row, cloudcover and processing state.
    
    scenes = s3_scenes[(s3_scenes.path == path) & (s3_scenes.row == row) & (s3_scenes.cloudCover <= 5) & (~s3_scenes.productId.str.contains('_T2')) & (~s3_scenes.productId.str.contains('_RT'))]
    
    print(' Found {} images\n'.format(len(scenes)))
                       
    # If any scenes exists, select the one that have the minimum cloudCover.
    if len(scenes):
        scene = scenes.sort_values('cloudCover').iloc[0]
    # Add the selected scene to the bulk download list.
    bulk_list.append(scene)


bulk_frame = pd.concat(bulk_list, 1).T

bulk_frame

# For each row
for i, row in bulk_frame.iterrows():
    # Print some the product ID
    print('\n', 'EntityId:', row.productId, '\n')
    print(' Checking content: ', '\n')
    # Request the html text of the download_url from the amazon server.
    # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
    response = requests.get(row.download_url)
    # If the response status code is fine (200)
    if response.status_code == 200:
        # Import the html to beautiful soup
        html = BeautifulSoup(response.content, 'html.parser')
        # Create the dir where we will put this image files.
        entity_dir = os.path.join(LANDSAT_PATH, row.productId)
        if not os.path.exists(entity_dir):
            os.makedirs(entity_dir)
        
        # Second loop: for each band of this image that we find using the html <li> tag
        for li in html.find_all('li'):
            # Get the href tag
            file = li.find_next('a').get('href')
            print('  Downloading: {}'.format(file))
            # Download the files
            # code from: https://stackoverflow.com/a/18043472/5361345
            response = requests.get(row.download_url.replace('index.html', file), stream=True)
            with open(os.path.join(entity_dir, file), 'wb') as output:
                shutil.copyfileobj(response.raw, output)
            del response

#Read the rasters using rasterio and extract the bounds.

xmin, xmax, ymin, ymax = [], [], [], []

for image_path in glob(os.path.join(LANDSAT_PATH, '*/*B10.TIF')):
    with rasterio.open(image_path) as src_raster:
        xmin.append(src_raster.bounds.left)
        xmax.append(src_raster.bounds.right)
        ymin.append(src_raster.bounds.bottom)
        ymax.append(src_raster.bounds.top)

fig, ax = plt.subplots(1, 1, figsize=(20, 15), subplot_kw={'projection': ccrs.UTM(16)})

ax.set_extent([min(xmin), max(xmax), min(ymin), max(ymax)], ccrs.UTM(16))

bounds.plot(ax=ax, transform=ccrs.PlateCarree())


for image_path in glob(os.path.join(LANDSAT_PATH, '*/*B10.TIF')):
    
    with rasterio.open(image_path) as src_raster:
        
        extent = [src_raster.bounds[i] for i in [0, 2, 1, 3]]
        
        dst_transform = from_origin(src_raster.bounds.left, src_raster.bounds.top, 250, 250)
        
        width = np.ceil((src_raster.bounds.right - src_raster.bounds.left) / 250.).astype('uint')
        height = np.ceil((src_raster.bounds.top - src_raster.bounds.bottom) / 250.).astype('uint')
        
        dst = np.zeros((height, width))
        
        reproject(src_raster.read(1), dst,
                  src_transform=src_raster.transform,
                  dst_transform=dst_transform,
                  resampling=Resampling.nearest)
                  
        ax.matshow(np.ma.masked_equal(dst, 0), extent=extent, transform=ccrs.UTM(16))


fig.savefig('./images/landsat.png')


#TOA
from glob import glob
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import rasterio

from l8qa.qa import write_cloud_mask
from rio_toa import reflectance

from tqdm import tqdm_notebook

from cartopy import crs as ccrs
import geopandas as gpd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

SRC_LANDSAT_FOLDER = './data/'
DST_LANDSAT_FOLDER = './data/'

creation_options = {'nodata': 0,
    'compress': 'deflate',
        'predict': 2}

processes = 4
rescale_factor = 55000
dtype = 'uint16'

# Use `glob` to find all Landsat-8 Image folders.
l8_images = glob(os.path.join(SRC_LANDSAT_FOLDER, 'L*/'))

# Here we will set up `tqdm` progress bars to keep track of our processing.
# (This is an optional step, I like progress bars)
pbar_1st_loop = tqdm_notebook(l8_images, desc='Folder')

for src_folder in pbar_1st_loop:
    # Get the name of the current folder
    print(src_folder)
    folder = os.path.split(src_folder[:-1])[-1]
    
    # Make a folder with the same name in the `DST_LANDSAT_FOLDER`.
    dst_folder = os.path.join(DST_LANDSAT_FOLDER, folder)
    os.makedirs(dst_folder, exist_ok=True)
    
    # Use `glob` again to find the metadata (`MTL.txt`) and the Quality Assessment (`QA`) files
    src_mtl = glob(os.path.join(src_folder, '*MTL*'))[0]
    src_qa = glob(os.path.join(src_folder, '*QA*'))[0]
    
    # Here we will create the cloudmask
    # Read the source QA into an rasterio object.
    with rasterio.open(src_qa) as qa_raster:
        # Update the raster profile to use 0 as 'nodata'
        profile = qa_raster.profile
        profile.update(nodata=0)
        
        #Call `write_cloud_mask` to create a mask where the QA points to clouds.
        write_cloud_mask(qa_raster.read(1), profile, os.path.join(dst_folder, 'cloudmask.TIF'))


# Set up the second loop. For bands 1 to 9 in this image.
pbar_2nd_loop = tqdm_notebook(range(1, 10), desc='Band')

# Iterate bands 1 to 0 using the tqdm_notebook object defined above
for band in pbar_2nd_loop:
    # Use glob to find the current band GeoTiff in this image.
    src_path = glob(os.path.join(src_folder, '*B{}.TIF'.format(band)))
    dst_path = os.path.join(dst_folder, 'TOA_B{}.TIF'.format(band))
        # Writing reflectance takes a bit to process, so if it crashes during the processing,
        # this code skips the ones that were already processed.
    if not os.path.exists(dst_path):
        # Use the `rio-toa` module for reflectance.
        reflectance.calculate_landsat_reflectance(src_path, src_mtl,dst_path, rescale_factor=rescale_factor, creation_options=creation_options,bands=[band], dst_dtype=dtype,processes=processes, pixel_sunangle=True)

# Just copy the metadata from source to destination in case we need it in the future.
shutil.copy(src_mtl, os.path.join(dst_folder, 'MTL.txt'))

from matplotlib.cm import viridis as cmap
import matplotlib.patheffects as pe

bounds = gpd.read_file('./data/area_of_study_bounds.gpkg')

xmin, xmax, ymin, ymax = [], [], [], []

for image_path in glob(os.path.join(DST_LANDSAT_FOLDER, '*/*B5.TIF')):
    with rasterio.open(image_path) as src_raster:
        xmin.append(src_raster.bounds.left)
        xmax.append(src_raster.bounds.right)
        ymin.append(src_raster.bounds.bottom)
        ymax.append(src_raster.bounds.top)

fig, ax = plt.subplots(1, 1, figsize=(20, 15), subplot_kw={'projection': ccrs.UTM(16)})
hist_fig, hist_axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

ax.set_extent([min(xmin), max(xmax), min(ymin), max(ymax)], ccrs.UTM(16))

bounds.plot(ax=ax, transform=ccrs.PlateCarree())

for hax, image_path in zip(hist_axes.ravel(), glob(os.path.join(DST_LANDSAT_FOLDER, '*/*B5.TIF'))):
    
    pr = image_path.split('/')[-2].split('_')[2]
    path_row = 'P: {}  R: {}'.format(pr[:3], pr[3:])
    
    with rasterio.open(image_path) as src_raster:
        
        extent = [src_raster.bounds[i] for i in [0, 2, 1, 3]]
        
        dst_transform = rasterio.transform.from_origin(src_raster.bounds.left, src_raster.bounds.top, 250, 250)
        
        width = np.ceil((src_raster.bounds.right - src_raster.bounds.left) / 250.).astype('uint')
        height = np.ceil((src_raster.bounds.top - src_raster.bounds.bottom) / 250.).astype('uint')
        
        dst = np.zeros((height, width))
        
        rasterio.warp.reproject(src_raster.read(1), dst,
                                src_transform=src_raster.transform,
                                dst_transform=dst_transform,
                                resampling=rasterio.warp.Resampling.nearest)
            
                                cax = ax.matshow(np.ma.masked_equal(dst, 0) / 550.0, cmap=cmap,
                                                 extent=extent, transform=ccrs.UTM(16),
                                                 vmin=10, vmax=40)
                                
                                t = ax.text((src_raster.bounds.right + src_raster.bounds.left) / 2,
                                            (src_raster.bounds.top + src_raster.bounds.bottom) / 2,
                                            path_row,
                                            transform=ccrs.UTM(16),
                                            fontsize=20, ha='center', va='top', color='.1',
                                            path_effects=[pe.withStroke(linewidth=3, foreground=".9")])
                                
                                sns.distplot(dst.ravel() / 550.0, ax=hax, axlabel='Reflectance (%)', kde=False, norm_hist=True)
                                hax.set_title(path_row, fontsize=13)
                                hax.set_xlim(0,100); hax.set_ylim(0,.1)
                                
                                [patch.set_color(cmap(plt.Normalize(vmin=10, vmax=40)(patch.xy[0]))) for patch in hax.patches]
                                [patch.set_alpha(1) for patch in hax.patches]
                                
        del dst

fig.colorbar(cax, ax=ax, label='Reflectance (%)')
fig.savefig('./images/11/band-5.png', transparent=True, bbox_inches='tight', pad_inches=0)
hist_fig.savefig('./images/11/hist-band-5.png', transparent=True, bbox_inches='tight', pad_inches=0)





