# Automated Sentinel-2 GEOBIA
##Introduction
This script automates the geographic object-based image analysis (GEOBIA) workflow for supervised land cover classification. 
It is designed to work on Sentinel-2 datasets downloaded from the European Space Agency (ESA) Copernicus Hub.

##Requirements
  * Windows (unlikely to work on Linux or macOS)
  * Anaconda (can be downloaded from https://www.anaconda.com/products/individual#Downloads)

 ## Setup
 1. Clone or download this repository (if downloading as a zip file, remember to unzip it before use)
 2. Launch Anaconda command prompt
 3. Create environment: `conda create -n auto-class python=3.8`
 4. Activate environment: `conda activate auto-class`
 5. Install dependencies: `conda env update -f environment.yml --prune`

##Quick start
1. Launch Anaconda command prompt
2. Activate the environment: `conda activate auto-class`
3. Navigate to the location where you cloned/extracted this repository: `cd path\to\this\repository`.
This should be the root directory of the repository where script.py is located.
   
4. Place the required datasets in the `input` folder (see **Data** section for details of required datasets)
5. Execute the script with default settings with `python script.py`
6. If successful, a new `output` folder should be created containing your classified image and a csv file containing the
new raster values of each land cover class.

##Data
The script requires a zip file of Sentinel-2 data, a shapefile of training data and (optionally but recommended)
a shapefile defining the extent of the output image. 

###Sentinel data
Sentinel-2 imagery can be downloaded from the ESA Copernicus Hub at https://scihub.copernicus.eu/dhus/#/home. 
A free account is required to download datasets. The downloaded zip file should be placed in the `input` folder. 
**Do not place more than one zip file in the input folder**.

**Note:** The script expects the imagery data as a zip file. **There is no need to unzip the contents of the file downloaded from
the Copernicus Hub.** You can of course rename the downloaded zip file or provide your own zip file of single-band rasters,
provided they are in .jp2 format and follow the Sentinel band naming convention (e.g. _B02 is the file name suffix for band 2).

###Included example data
The included example data `.\input\extent.shp` and `.\input\training_data.shp` are designed to be used with 
Sentinel-2 imagery with a relative orbit number of **123**. 
The dataset used to test the script can be downloaded from 
https://scihub.copernicus.eu/dhus/odata/v1/Products('494c92e2-90ed-4b43-aece-cdcd6ee36859')/$value.

###Training data
The training data shapefile should contain only point data with the land cover class names stored as text (string) in the attribute table.
By default, the column containing the class names is expected to be called 'Ecological'. A different column name should be defined
with the optional argument `--class_col` when executing the script. The training data should be located at `.\input\training_data.shp`.

###Defining an extent
Processing an entire Sentinel-2 image can take a very long time, so it is recommended to define a processing extent
that is smaller than the full extent of the image with an extent shapefile. 
This shapefile should contain a single polygon that is fully within the bounds of the input imagery.
By default, the extent shapefile should be located at `.\input\extent.shp`.

##Optional arguments
The script has a number of optional arguments that can be defined when executing the script from the command line.

For example: `python script.py -i different\input\folder -o different\output\folder --class_col 'class_name' --validate`

Help for these arguments can be accessed with `python script.py -h` but is also provided below.

###Custom folder paths
These options are used to replace the default locations of the input, output and temporary folders used by the script.
Each of these arguments should be followed by a custom folder path.

`--input_path` or `-i` sets a custom `input` folder. This is the folder containing your image data zip file, 
training data shapefile and extent shapefile.
`--output_path` or `-o` sets a custom `output` folder. This is where the classified image and class csv file will be placed.

