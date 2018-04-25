#!/bin/bash                                                                                                       

filename="content.tsv"
hdr=$"index\twebsite_name\tad_type\turl\ttext\tinfo_extracted"
newdir="splitted_files_ver1"
mkdir $newdir
split -l 5000000 $filename $newdir/split
n=1
for f in $newdir/split*
do
echo $f
echo -e $hdr > $newdir/Part${n}.tsv
cat $f >> $newdir/Part${n}.tsv
((n++))
rm $f
done

