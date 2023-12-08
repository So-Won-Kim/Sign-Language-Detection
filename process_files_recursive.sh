#!/bin/bash

process_jpg() {
    echo "Processing file: $1"
}

jpg_directory="./"

for jpg_file in "$jpg_directory"/*.jpg; do
    if [ -f "$jpg_file" ]; then
        python3 get_letter.py "$jpg_file"
    fi
done
