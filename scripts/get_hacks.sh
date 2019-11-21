#!/usr/bin/env bash

# Download a bunch of SMW hacks.
#
# This is a list of hacks collected from SMW Central
# (https://smwcentral.net/) where the following criteria were used:
#    - Must have 'vanilla' tag.
#    - Must have a rating greater than or equal to 3.0.
# Duplicate or faulty patches have been removed; the least customized or
# English version was kept.
#
# For portability, it is possible to download an archive that is over
# twice as large but uses tar (and gzip) instead of 7-Zip.

p7z_bin="7z"

hacks_dl="https://drive.google.com/uc?export=download&id=1WtCHPaA1tMHLGIaf6JOK0DpMbyXpr6rw"
hacks_tar_dl="https://drive.google.com/uc?export=download&id=1-ZznV5ignf8I1aOEJnMC7wZ7baH3E6ls"

print_usage() {
    echo "Usage: $0 [-k] [-t] [-z <7z_bin>] <parent_dir>"
    echo ""
    echo "Required arguments:"
    echo "   parent_dir: Where to save the downloaded hacks."
    echo ""
    echo "Optional arguments:"
    echo "   -k:        Keep the archive, do not delete it after extraction."
    echo "   -t:        Use tar instead of 7-Zip (much larger download)."
    echo "   -z 7z_bin: Path to a 7-Zip (or p7zip) binary. Default: 7z."
    echo "   -h:        Print this help message."
}


# Download the given URL to the given destination.
#    $1: A URL containing an archive of hacks.
#    $2: Download destination.
download_hacks() {
    echo "Downloading all hacks..."
    wget -qO $2 "$1"
    echo "Done downloading."
}


# Unzip the given file to the given directory keeping old files.
#    $1: Directory to unzip to.
unzip_hacks() {
    echo "Unzipping hacks..."
    $p7z_bin x -aos -o"$1" hacks.7z > /dev/null
}


# Untar the given file to the given directory keeping old files.
#    $1: Directory to untar to.
untar_hacks() {
    echo "Untaring hacks..."
    tar -xzkf hacks.tar.gz -C "$1"
}


keep_zips=''
use_tar=''
archive_name="hacks.7z"

while getopts ":ktz:h" opt; do
    case "$opt" in
        k)
            keep_zips=t
            ;;
        t)
            use_tar=t
            archive_name="hacks.tar.gz"
            ;;
        z)
            p7z_bin="${OPTARG}"
            ;;
        h)
            print_usage
            exit 0
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

parent_dir=$1

# Check for missing inputs.
if [ -z "$parent_dir" ]; then
    print_usage
    exit 1
fi

if [ "x$use_tar" != x ]; then
    download_hacks "$hacks_tar_dl" $archive_name
    untar_hacks $parent_dir
# Check for 7-Zip
elif ! [ -x "$(command -v $p7z_bin)" ]; then
    echo "7-Zip ($p7z_bin) could not be found."
    echo "Please use the -t option or make sure it is in your PATH."
    exit 1
else
    download_hacks "$hacks_dl" $archive_name
    unzip_hacks $parent_dir
fi

if [ "x$keep_zips" != x ]; then
    mv $archive_name $parent_dir
else
    rm $archive_name
fi

echo "Done."

