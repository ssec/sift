#!/usr/bin/env bash

script_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

input_dir=$1
output_dir=$2

if [ $# -ne 2 ]; then
    echo "Usage: ./go.sh input_dir output_dir"
    exit 1
fi

oops() {
    echo "ERROR: $@"
    exit 1
}

abspath() {
    python -c "import os; print(os.path.realpath(\"$1\"))"
}

if [ ! -d $input_dir ]; then
    echo "Input directory does not exist: $input_dir"
    exit 1
fi

# Get full path
input_dir=`abspath $input_dir`

if [ ! -d $output_dir ]; then
    echo "Output directory does not exist: $output_dir"
    echo "Creating output directory: $output_dir"
    mkdir -p $output_dir || oops "Could not create output directory"
fi

cd $output_dir

#OUTPUT_SIZE_X=8192
#OUTPUT_SIZE_Y=4096
OUTPUT_RES_X=2000
OUTPUT_RES_Y=2000

for nc_file in `find $input_dir/ -name "*.nc"`; do
    echo "Processing $nc_file..."
    base_fn=`basename $nc_file`
    geos_gtiff_fn=${base_fn/.nc/.tif}
    merc_gtiff_fn=${base_fn/.nc/.merc.tif}

    echo "Creating GEOS geotiff file..."
    time python $script_dir/ahi2gtiff.py $nc_file -o $geos_gtiff_fn
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create geos geotiff"
        continue
    fi

    echo "Creating mercator geotiff file..."
    time gdalwarp -multi -t_srs "+proj=merc +datum=WGS84 +ellps=WGS84" -tr $OUTPUT_RES_X $OUTPUT_RES_Y -te_srs "+proj=latlong +datum=WGS84 +ellps=WGS84" -te -180 -80 180 80 $geos_gtiff_fn $merc_gtiff_fn
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create mercator geotiff"
        continue
    fi
done
