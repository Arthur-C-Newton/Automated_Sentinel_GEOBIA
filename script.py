# This is the main script
# the docstring should go here

# import the required modules
from zipfile import ZipFile
import os
from pathlib import Path
import rasterio
from rasterio import mask
from rasterio.windows import get_data_window
import geopandas as gpd
import gdal
from rsgislib.segmentation import segutils
import rsgislib.rastergis
import rsgislib.rastergis.ratutils

# file paths and other elements that should be set by user
extent_path = "input\\extent.shp"  # replace these paths later to allow users to set this when running script
tmp_path = ".\\tmp"
stack_path = "tmp\\stack.tif"  # users should only want to change this if they want to only save the stacked image
shp_path = "input\\training_data.shp"

for file in os.listdir(".\\input"):
    if file.endswith(".zip"):
        zip_path = os.path.join(".\\input", file)

class_col_name = 'Ecological'  # this is the name of the column of the training data that contains class strings

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
meta.update(count=len(bands), driver="GTiff")  # update the metadata to allow multiple bands

# clip to extent if one is provided
if os.path.exists(extent_path):
    extent = gpd.read_file(extent_path)
    shapes = extent["geometry"]
    clipped_image, clipped_transform = mask.mask(band2, shapes, crop=True)
    clipped_meta = meta
    clipped_meta.update({"driver": "GTiff",
                         "height": clipped_image.shape[1],
                         "width": clipped_image.shape[2],
                         "transform": clipped_transform})
    # create a single stacked geotiff based on the image metadata
    with rasterio.open(stack_path, 'w', **clipped_meta) as dst:
        for id, layer in enumerate(bands, start=1):
            with rasterio.open("zip:" + zip_path + "!" + layer) as src1:
                out_image, transform = mask.mask(src1, shapes, crop=True)
                dst.write_band(id, out_image[0])
                print("Writing band...")
else:  # if no extent is provided, the full image is staved to disk (very slow)
    # create a single stacked geotiff based on the image metadata
    with rasterio.open(stack_path, 'w', **meta) as dst:
        for id, layer in enumerate(bands, start=1):
            with rasterio.open("zip:" + zip_path + "!" + layer) as src1:
                dst.write_band(id, src1.read(1))
                print("Writing band...")
print("Multi-band GeoTiff saved successfully at tmp/stack.tif!")

# re-import data and save as KEA
raster = gdal.Open(stack_path)
raster = gdal.Translate("tmp\\raster.kea", raster, format="KEA")

in_img = "tmp\\raster.kea"
clumps = "clumps_image.kea"


# segment the image using rsgislib
segutils.runShepherdSegmentation(in_img, clumps, tmpath=tmp_path, numClusters=100, minPxls=100, distThres=100, sampling=100, kmMaxIter=200)
band_info = []
band_info.append(rsgislib.rastergis.BandAttStats(band=1, minField='BlueMin', maxField='BlueMax', meanField='BlueMean', stdDevField='BlueStdev'))
band_info.append(rsgislib.rastergis.BandAttStats(band=2, minField='GreenMin', maxField='GreenMax', meanField='GreenMean', stdDevField='GreenStdev'))
band_info.append(rsgislib.rastergis.BandAttStats(band=3, minField='RedMin', maxField='RedMax', meanField='RedMean', stdDevField='RedStdev'))
band_info.append(rsgislib.rastergis.BandAttStats(band=4, minField='NIRMin', maxField='NIRMax', meanField='NIRMean', stdDevField='NIRStdev'))
rsgislib.rastergis.populateRATWithStats(in_img, clumps, band_info)


# import training data from shapefile
training_data = gpd.read_file(shp_path)
print(training_data.head())  # for testing

# save each land cover class as a separate shapefile
for l_class, training_class in training_data.groupby(class_col_name):
    training_class.to_file(tmp_path + "\\" + l_class + ".shp")

# read shapefiles of each class into dictionary
class_dict = dict()
index = 0
for file in os.listdir(tmp_path):
    if file.endswith(".shp"):
        class_path = os.path.join(tmp_path, file)
        class_filename = Path(class_path).stem
        class_name_nospace = class_filename.replace(" ", "_")
        index += 1
        class_dict[class_name_nospace] = [index, class_path]

class_int_col_in = "ClassInt"
class_name_col = "ClassStr"
rsgislib.rastergis.ratutils.populateClumpsWithClassTraining(clumps, class_dict, tmp_path, class_int_col_in, class_name_col)
