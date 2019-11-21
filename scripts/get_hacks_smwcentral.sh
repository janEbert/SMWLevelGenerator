#!/usr/bin/env bash

# Download a ton of SMW hacks. To not overload the SMW Central website,
# a 30 sec timeout is awaited after each download.
#
# Seems to not work anymore with SMW Central's new DDoS detection
# service. :(

# TODO implement rating filter


print_usage() {
    echo "Usage: $0 [-g <html_file>] [-p <html_file>] <in_file> <dl_dir>"
    echo ""
    echo "Required arguments:"
    echo "   in_file: Parsed level list."
    echo "   dl_dir:  Where to save the downloaded hacks."
    echo ""
    echo "Optional arguments:"
    echo "   -g html_file: (Re-)generate the hack list and write it to"
    echo "                 the given file."
    echo "   -p html_file: Parse the given hack list. Writes to in_file (see above)."
    echo "                 Can be an empty string if -g is given."
    echo "   -e html_file: Print the amount of level exits in the given hack list."
    echo "                 Do not download. Can be an empty string if -g or -p is given."
    echo "   -h:           Print this help message."
}


# Get the amount of level pages in the given file.
#    $1: An HTML level list file.
get_num_pages() {
    grep -oP '(?<=<a href="/\?p=section&amp;s=smwhacks&amp;u=0&amp;g=0&amp;f\[tags]=%2Bvanilla&amp;n=)\d+(?=&amp;o=rating&amp;d=desc" title="Go to page )' \
            $1 \
    | awk 'BEGIN { max=1 } { if ($1 > max) { max=$1 } } END { print max }'
}


# Write the list of levels to the given file.
# This only needs to be called once or in case you want to re-generate.
# Will delete contents of the given file.
#    $1: Where to write to (.html file).
generate_html_list() {
    echo "Generating HTML..."
    wget -qO $1 "https://www.smwcentral.net/?p=section&s=smwhacks&u=0&g=0&f[tags]=%2Bvanilla&n=1&o=rating&d=desc"
    lastpage=$(get_num_pages $1)
    for i in $(seq $lastpage); do
        wget -w 5 -q -O - "https://www.smwcentral.net/?p=section&s=smwhacks&u=0&g=0&f[tags]=%2Bvanilla&n=$i&o=rating&d=desc" \
        >> $1
    done
    echo "Done generating."
}


# Write the parsed list of download links from the given file to another
# given file.
#    $1: An HTML level list file.
#    $2: A file to write the parsed links to.
html_to_links() {
    # TODO maybe write awk script to be able to get rating and amount of exits
    echo "Parsing HTML..."
    grep -oP '(?<=href=")//dl.*(?=">Download<)' $1 \
    | awk '{print "https:" $1}' \
    | xargs -n 1 echo > $2
    echo "Done parsing."
}


# Print the amount of exits in the given file.
#    $1: An HTML level list file.
print_exits() {
    grep -oP '(?<=\t).*?(?= exit)' $1 \
    | awk 'BEGIN { i=0 } { i+=$1 } END { print i }'
}


# Download all the hacks in the given list of links to the given
# directory waiting a little between each download.
#    $1: File containing a list of links.
#    $2: Directory to download to.
download_hacks() {
    echo "Starting to download all hacks..."
    wget -w 30 -P $2 -i $1
    echo "Done."
}


generate=''
html_file=''
parse=''
parse_html_file=''
print_exits=''
exits_html_file=''

while getopts ":g:p:e:h" opt; do
    case "$opt" in
        g)
            generate=t
            html_file="${OPTARG}"
            ;;
        p)
            parse=t
            parse_html_file="${OPTARG}"
            ;;
        e)
            print_exits=t
            exits_html_file="${OPTARG}"
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

in_file=$1
dl_dir=$2

# Check for missing inputs.
if [ -z "$in_file" ] || [ -z "x$dl_dir" ] || \
        ([ "x$generate" != x ] && [ -z "$html_file" ]); then
    print_usage
    exit 1
fi

if [ "x$parse" != x ] && [ -z "$parse_html_file" ]; then
    if [ "x$html_file" != x ]; then
        parse_html_file=$html_file
    else
        print_usage
        exit 1
    fi
fi

if [ "x$print_exits" != x ] && [ -z "$exits_html_file" ]; then
    if [ "x$html_file" != x ]; then
        exits_html_file=$html_file
    elif [ "x$parse_html_file" != x ]; then
        exits_html_file=$parse_html_file
    else
        print_usage
        exit 1
    fi
fi

if [ "x$generate" != x ]; then
    generate_html_list $html_file
fi

if [ "x$parse" != x ]; then
    html_to_links $parse_html_file $in_file
fi

if [ "x$print_exits" != x ]; then
    print_exits $exits_html_file
    exit 0
fi

download_hacks "$in_file" "$dl_dir"

