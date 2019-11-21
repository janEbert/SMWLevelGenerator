#!/usr/bin/env bash

# Dump the levels contained in the ROMs contained in the
# given directory.

exec_dir=$(pwd -P)
parent_dir=$(cd "$(dirname $0)"; pwd -P)

wine_bin="wine"

lunar_magic="$parent_dir/../tools/lm304ebert21892usix/Lunar Magic.exe"


print_usage() {
    echo "Usage: $0 [-h] [-w <wine_bin>] <in_dir> <out_dir>"
    echo "Required arguments:"
    echo "   in_dir:  Directory containing all the .smc ROM files."
    echo "   out_dir: Where to dump the level files."
    echo ""
    echo "Optional arguments:"
    echo "   -w wine_bin: Path to a Wine binary. Default: wine."
    echo "   -h:          Print this message."
}


# Dump all levels in the ROMs in the given directory to the
# given directory.
#    $1: The directory containing the .smc files.
#    $2: Where to dump the levels to.
dump_levels() {
    echo "Dumping levels..."
    find "$1" -type f -name "*.smc" \
    | while read rom; do
        rom_basename=$(basename "$rom" .smc)
        echo "Processing $rom_basename.smc..."
        level_dir="$2/$rom_basename"
        mkdir -p "$level_dir"
        cd "$level_dir"
        # We link here because Lunar Magic dumps to where the given
        # ROM lies.
        ln -sf "$exec_dir/$rom" "$rom_basename.smc"
        $wine_bin "$lunar_magic" -DeconstructLevel "$rom_basename.smc"
        $wine_bin "$lunar_magic" -ExportMap16FG "$rom_basename.smc" "map16fg.bin" 105
        # Don't forget to clean up after yourself! (Remove the link.)
        # Or do forget because you are going to use it again.
        # rm "$rom_basename.smc"
        cd "$exec_dir"
    done
    echo "Done."
}


while getopts ":w:h" opt; do
    case "$opt" in
        w)
            wine_bin="${OPTARG}"
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

in_dir=$1
out_dir=$2

# Check for Wine (Lunar Magic does not have Linux binaries)
if ! [ -x "$(command -v $wine_bin)" ]; then
    echo "Wine ($wine_bin) could not be found."
    echo "Please make sure it is installed and in your PATH."
    echo "Check https://www.winehq.org/ for information."
    exit 1
fi

if [ -z "$in_dir" ] || [ -z "$out_dir" ]; then
    print_usage
    exit 1
fi

dump_levels "$in_dir" "$out_dir"

