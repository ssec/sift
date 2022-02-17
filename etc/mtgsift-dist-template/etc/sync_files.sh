#!/usr/bin/env bash

# sync files from a source dir into a remote dest dir. Files will only be transfered once (files delete from the destination are not re-transfered) and files deleted from the source are
# not deleted from the destination

set -x
set -e

track_dir=/opt/mtg-sift/track
inc_file="$track_dir/include_files"
exc_file="$track_dir/exclude_files"

#hrit
src_dir="/tcenas/fbf/EUMETCAST/in/TCE_bas/EUMETSAT_Data_Channel_2"
#seviri native
src_dir="/tcenas/fbf/MSG/in/MSG4/OPE4/SEVI-MSG15"
#src_dir="/opt/mtg-sift/Data/MSG-2"
dst_dir="/opt/mtg-sift/Data/MSG-Incoming"

mkdir -p "$track_dir"
touch $exc_file

current_date_time="`date +%Y%m%d%H%M%S`";
echo "$current_date_time: look for files in src dir"

#list=`find $src_dir -type f -cmin -60 \( -name "*IR_108*" -o -name "*EPI*" -o -name "*PRO*" \) -printf "%T+ %p\n" | sort`
list=`find $src_dir -type f -cmin -60 -name "*.nat" -printf "%T+ %p\n" | sort`

#create an empty inc_file
echo "" > $inc_file


#Set the field separator to new line
IFS=$'\n'

for a_file in $list
do
   fname=`echo $a_file | cut -d' ' -f2`

   echo $(basename $fname) >> $inc_file
done

#rsync -va --dry-run --delete --exclude-from "$exc_file" --include-from "$inc_file" "$src_dir" "$dst_dir"
rsync -va --exclude-from "$exc_file" --files-from "$inc_file" "$src_dir" "$dst_dir"

current_date_time="`date +%Y%m%d%H%M%S`";
echo "$current_date_time: Start ftp transfer"

# Add the included/transferred files to the exclusion list
cat "$inc_file" "$exc_file" > "$exc_file".tmp
sort "$exc_file".tmp | uniq > "$exc_file"

