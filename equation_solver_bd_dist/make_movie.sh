#!/bin/bash

DIR=$1

# Ensure DIR exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist"
    exit 1
fi

# Generate frames
python plot_frames.py --dir "$DIR"

cd "$DIR"

# Make movie
ffmpeg -framerate 15 -i frames/frame_%04d.png \
       -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
       -c:v libx264 -pix_fmt yuv420p movie.mp4

# Optional: remove frames
rm -r frames/
