#!/bin/bash

fname=$1
if [[ $fname =~ "_fonts_embedded.pdf" ]]
then
    exit 0
else
    outname=`basename $fname .pdf`_fonts_embedded.pdf
    gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -sOutputFile=$outname -c ".setpdfwrite <</NeverEmbed [ ]>> setdistillerparams" -f $fname
fi
