#!/usr/bin/env bash

# Unzip all hacks in the given directory to the other given directory.


print_usage() {
    echo "Usage: $0 [-h] <zips_dir> <out_dir>"
    echo "Required arguments:"
    echo "   zips_dir: Directory containing the compressed hacks."
    echo "   out_dir:  Where to extract to."
    echo ""
    echo "Optional arguments:"
    echo "   -h: Print this message."
}


# Unzip all hacks in the given directory to the other given directory.
#    $1: The input directory of zip files.
#    $2: The output directory of extracted files.
unzip_hacks() {
    mkdir "$2"
    # We do _not_ want child directories as well!
    echo "Extracting hacks..."
    ls "$1" \
    | sed "/\.zip$/ s/\.zip$//" \
    | xargs -d '\n' -P 8 -n 1 -I % unzip -oqq "$1/%.zip" -d "$2/%"
    echo "Done."
}


if [ "x$1" = "x-h" ] || [ "x$1" = "x--help" ]; then
    print_usage
    exit 0
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    print_usage
    exit 1
fi

unzip_hacks "$1" "$2"

