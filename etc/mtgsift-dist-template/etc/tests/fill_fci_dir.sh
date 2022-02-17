#!/bin/bash

OUTPUT_DIR=/opt/mtg-sift/Data/MTG-Incoming
echo "Clean output dir"
rm -Rf $OUTPUT_DIR/*.nc

/opt/mtg-sift/mtgsift-0.8/etc/tests/py-tests/bin/python3 fill_dir_periodically.py -i /tcenas/proj/optcalimg/test_data/SIFT/fci_l1c_fdhsi/26h_dataset/20200404_RC139 -o /opt/mtg-sift/Data/MTG-Incoming -t 20
