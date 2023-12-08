#!/bin/bash

process_jpg() {
    echo "Processing file: $1"
}

jpg_directory="path/to/output_directory"

for jpg_file in "$jpg_directory"/*.jpg; do
    if [ -f "$jpg_file" ]; then
        process_jpg "$jpg_file"
    fi
done
