CSPOV
=====

TODO

Reproject AHI Recipe
--------------------

python3 py/cspov/project/ahi2gtiff.py HS_H08_20150714_0330_B01_FLDK_R20.nc
# creates HS_H08_20150714_0330_B01_FLDK_R20.tif in ~2.5 seconds
gdalwarp -t_srs "+proj=merc +datum=WGS84 +ellps=WGS84" -ts 5120 5120 -te_srs "+proj=latlong +datum=WGS84 +ellps=WGS84" -te -180 -80 180 80 HS_H08_20150714_0330_B02_FLDK_R20{,.merc}.tif
# creates HS_H08_20150714_0330_B01_FLDK_R20.merc.tif in 20-30 seconds
# created mercator geotiffs are 8-bit unsigned integers, -180 to 180 longitude and -80 to 80 latitude

# Or for multiple files in a directory
for fn in *.nc; do time python ~/repos/git/CSPOV/py/cspov/project/ahi2gtiff.py -vvv $fn; done

for fn in *R20.tif; do
    time gdalwarp -t_srs "+proj=merc +datum=WGS84 +ellps=WGS84" -ts 5120 5120 -te_srs "+proj=latlong +datum=WGS84 +ellps=WGS84" -te -180 -80 180 80 $fn ${fn/.tif/.merc.tif}
done