#!/usr/bin/env sh

# Download and extract the toolchain consisting of a custom-built
# Lunar Magic and Floating IPS.

# Required tools
required_tools_dl="https://drive.google.com/uc?export=download&id=1Xa93mzV--rX0S-0X7euB3fmrJRrkCUv6"
lunar_magic="lm304ebert21892usix.zip"
flips="floating.zip"
required_tools="$lunar_magic\n$flips"

# Optional tools
optional_tools_dl="https://drive.google.com/uc?export=download&id=19lmyH1xntGXIhp77mqvH9_djgXSrRhQs"
lmsw="lmsw.zip"
optional_tools="$lmsw"

all_tools="$required_tools\n$optional_tools"


print_usage() {
    echo "Usage: $0 [-k] [-e] [-h] dl_dir"
    echo ""
    echo "Required arguments:"
    echo "   dl_dir: Where to download the tools to."
    echo ""
    echo "Optional arguments:"
    echo "   -k: Keep the zips."
    echo "   -e: Get extra optional tools."
    echo "   -h: Print this message."
}


# Download and unzip the file in the given URL to the given directory.
#    $1: URL to download from.
#    $2: Download destination directory.
download_unzip() {
    dl_dir="$2"
    tools_zip="$dl_dir/tools.zip"
    mkdir -p $dl_dir
    wget -qO $tools_zip "$1"
    unzip -oqq "$tools_zip" -d $dl_dir
    rm "$tools_zip"
}


# Download required tools to the given directory.
#    $1: Where to save the tools.
download_tools() {
    echo "Downloading required tools..."
    download_unzip "$required_tools_dl" $1
}


# Download optional tools to the given directory.
#    $1: Where to save the tools.
download_extras() {
    echo "Downloading optional tools..."
    download_unzip "$optional_tools_dl" $1
}


# Unzip all tools in the given directory, each to a new directory.
#    $1: Directory containing the zips.
unzip_tools() {
    echo "$all_tools" \
    | while read zip; do
        file="$1/$zip"
        if [ -e $file ]; then
            if [ "$file" = "$lmsw" ]; then
                # LMSW should be unzipped into the Lunar Magic directory
                unzip -oq $file -d "$1/$(basename $lunar_magic .zip)"
            else
                unzip -oq $file -d "$1/$(basename $file .zip)"
            fi
        fi
    done
}


# Move all zips to a 'zips' directory in the given directory.
#    $1: Directory containing .zip files.
move_zips() {
    mkdir -p "$1/zips"
    echo "$all_tools" \
    | while read zip; do
        mv "$1/$zip" "$1/zips/"
    done
}


# Remove all downloaded zip files in the given directory.
#    $1: Directory containing .zip files.
remove_zips() {
    echo "$all_tools" \
    | while read file; do
        rm $1/$file
    done
}


keep_zips=''
get_extras=''

while getopts ":keh" opt; do
    case "$opt" in
        k)
            keep_zips=t
            ;;
        e)
            get_extras=t
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

dl_dir=$1

if [ -z "$dl_dir" ]; then
    print_usage
    exit 1
fi

download_tools $dl_dir

if [ "x$get_extras" != x ]; then
    download_extras $dl_dir
fi

echo "Done downloading."

unzip_tools $dl_dir

if [ "x$keep_zips" != x ]; then
    move_zips $dl_dir
else
    remove_zips $dl_dir
fi

echo "Done."

