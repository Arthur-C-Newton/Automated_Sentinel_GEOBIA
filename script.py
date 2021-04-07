# This is the main script
# the docstring should go here

# import the required modules
from zipfile import ZipFile
import os
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

stack_path = "tmp\\stack.tif"
if not stack_path.exists():
    # find zip file in input folder
    for file in os.listdir(".\\input"):
        if file.endswith(".zip"):
            zip_path = os.path.join(".\\input", file)

    # create ist of files that are band images
    archive = ZipFile(zip_path, 'r')
    files = [name for name in archive.namelist() if name.endswith('.jp2') and '_B' in name]

    # get the indices of the useful bands (B, G, R, NIR)
    index_b2 = [i for i, s in enumerate(files) if '_B02' in s]
    index_b3 = [i for i, s in enumerate(files) if '_B03' in s]
    index_b4 = [i for i, s in enumerate(files) if '_B04' in s]
    index_b8 = [i for i, s in enumerate(files) if '_B08' in s]

    # create a list of only the desired bands
    indices = [index_b2[0], index_b3[0], index_b4[0], index_b8[0]]
    bands = [files[i] for i in indices]  # The original band numbers are not preserved

    # read the metadata for the first image
    band2 = rasterio.open("zip:" + zip_path + "!" + files[index_b2[0]])
    meta = band2.meta
    meta.update(count=len(bands))  # update the metadata to allow multiple bands

    # create a single stacked geotiff based on the image metadata
    with rasterio.open('tmp\\stack.tif', 'w', **meta) as dst:
        for id, layer in enumerate(bands, start=1):
            with rasterio.open("zip:" + zip_path + "!" + layer) as src1:
                dst.write_band(id, src1.read(1))
                print("Writing band...")
    print("Multi-band GeoTiff saved successfully at tmp/stack.tif!")

# import training data
# find shapefiles in the input folder
for file in os.listdir(".\\input"):
    if file.endswith(".shp"):
        shp_path = os.path.join(".\\input", file)

training_data = gpd.read_file(shp_path)
print(training_data.head())
