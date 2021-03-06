#!/usr/bin/env bash

# Setup the environment, that is get all the necessary tools and hacks
# and generate all the levels contained in the hacks.
#
# You still have to obtain a clean Super Mario World ROM (as a
# .smc file) before you can execute this script.
# Check your ROM with this tool:
# https://media.smwcentral.net/onlinetools/jsromclean.htm
#
# The default structure created is the following:
# .
# |-- ... (all other folders already present in the repository)
# |-- compressed_hacks (all downloaded hacks)
# |-- hacks (uncompressed hacks)
# |-- levels (all the level data)
# |-- scripts (<- you are here; the download list of hacks will also be
# |            created here)
# |-- roms (all roms created by applying the hacks to the original game)
# `-- tools (all required and optional tools)

# TODO maybe give flags for cleanup and optional tools. at the moment,
# we are very liberal with space, but this is also supposed to be a
# simple script

parent_dir=$(dirname $0)

tools_dir="$parent_dir/../tools"
# A directory called 'hacks' will be created here.
hacks_parent_dir="$parent_dir/.."
hacks_dir="$hacks_parent_dir/hacks"
roms_dir="$parent_dir/../roms"
levels_dir="$parent_dir/../levels"

print_usage() {
    echo "Usage: $0 <smc_file>"
    echo ""
    echo "Required arguments:"
    echo "   smc_file: A .smc ROM file of Super Mario World (read this file's first"
    echo "             comment for detailed information)."
}

if [ "x$1" = "x-h" ]; then
    print_usage
    exit 0
fi

if [ -z "$1" ]; then
    print_usage
    exit 1
fi

sh $parent_dir/get_tools.sh -ek "$tools_dir"
sh $parent_dir/get_hacks.sh -k "$hacks_parent_dir" \
|| (echo "Using tar instead of 7z..."; \
sh $parent_dir/get_hacks.sh -kt "$hacks_parent_dir")
sh $parent_dir/create_roms.sh "$hacks_dir" "$roms_dir" "$1"
sh $parent_dir/dump_levels.sh "$roms_dir" "$levels_dir" \
|| (echo "Please manually execute dump_levels.sh after installing Wine."; \
echo "The full command is \`sh $parent_dir/dump_levels.sh $roms_dir $levels_dir\`")

echo "Done! Your environment is all setup."

