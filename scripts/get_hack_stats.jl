#!/usr/bin/env julia

"Collect and write various statistics about all hacks to a given file."
module GetHackStats

using Dates: DateTime, DateFormat

using CSV: write
using DataFrames: DataFrame


const statsdir  = abspath(joinpath(@__DIR__, "..", "stats"))
const hack_list = joinpath(statsdir, "hack_list.html")
const outfile   = joinpath(statsdir, "hack_stats.csv")

const DownloadsType = Int32
const LinkType = String
const SizeType = Int32
const RatingType = Union{Float32, Missing}
const AuthorsType = Vector{String}
const DifficultyType = String
const ExitsType = Int32
const FeaturedType = Bool
const DemoType = Bool
const DateType = DateTime
const IdType = String
const NameType = String


function getstats(file, asdict::Bool=false)::Union{DataFrame,Dict}
    getstats(file, Val(asdict))
end

function getstats(file, asdict::Val{false})::DataFrame
    stats = DataFrame(
            Downloads  = DownloadsType[],
            Link       = LinkType[],
            Size       = SizeType[],
            Rating     = RatingType[],
            Authors    = AuthorsType[],
            Difficulty = DifficultyType[],
            Exits      = ExitsType[],
            Featured   = FeaturedType[],
            Demo       = DemoType[],
            Date       = DateType[],
            Id         = IdType[],
            Name       = NameType[]
    )
    last_twenty = fill("n", 20)

    downloads_re  = r">([0-9,]+) downloads<"
    link_re       = r"<a href=\"(//dl\.smw.*?)\""
    size_re       = r"\t\t\t([0-9.]+ [KM])iB\t\t</td>"
    rating_re     = r"\t\t\t([0-9.]+)\t\t</td>"
    authors_re    = r"\t\t\t([^<].*)\t\t"
    authors_re2   = r"class=\"un\">(.*?)</a>"
    authors_re3   = r"class=\"un\">(.*?)</a></span>, ([^<].*?)</span>"
    difficulty_re = r"\t\t\t(.*?)\t\t</td>"
    exits_re      = r"\t(\d+) exit"
    featured_re   = r"\t\t\t(Yes|No)\t\t</td>"
    demo_re       = r"\t\t\t(Yes|No)\t\t</td>"
    date_re       = r"\t\t\t\t\t\t\t<span class=\"gray small\">Added: <time datetime=\"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\">.{22}</time></span>"
    idname_re     = r"id=(\d+)\">(.*?)</a>"

    date_format = DateFormat("y-m-dTH:M:S")

    for line in eachline(file)
        last_twenty = vcat(last_twenty[2:end], line)
        downloads_match = match(downloads_re, line)
        if !isnothing(downloads_match)
            linkline       = last_twenty[end - 1]
            sizeline       = last_twenty[end - 3]
            ratingline     = last_twenty[end - 5]
            authorline     = last_twenty[end - 7]
            difficultyline = last_twenty[end - 9]
            exitsline      = last_twenty[end - 11]
            featuredline   = last_twenty[end - 13]
            demoline       = last_twenty[end - 15]
            dateline       = last_twenty[end - 18]
            idnameline     = last_twenty[end - 19]
            rating_match = match(rating_re, ratingline)
            idname_match = match(idname_re, idnameline)

            push!(stats, (
                    parse(DownloadsType, replace(downloads_match.captures[1], ',' => "")),
                    "https:" * match(link_re, linkline).captures[1],
                    process_size(sizeline, size_re),
                    isnothing(rating_match) ? RatingType.a()
                            : parse(RatingType.b, rating_match.captures[1]),
                    process_authors(authorline, authors_re, authors_re2, authors_re3),
                    DifficultyType(match(difficulty_re, difficultyline).captures[1]),
                    parse(ExitsType, match(exits_re, exitsline).captures[1]),
                    FeaturedType(match(featured_re, featuredline).captures[1] == "Yes"),
                    DemoType(match(demo_re, demoline).captures[1] == "Yes"),
                    DateType(match(date_re, dateline).captures[1], date_format),
                    IdType(idname_match.captures[1]),
                    NameType(replace(replace(idname_match.captures[2],
                                             "&quot;" => '"'), "&amp;" => '&'))
            ))

        end
    end
    return stats
end

function getstats(file, asdict::Val{true})::Dict
    stats = Dict{String, Vector{String}}(stat => String[] for stat in (
            "downloads",
            "link",
            "size",
            "rating",
            "authors",
            "difficulty",
            "exits",
            "featured",
            "demo",
            "date",
            "id",
            "name"
    ))
    last_twenty = fill("n", 20)

    downloads_re  = r">([0-9,]+) downloads<"
    link_re       = r"<a href=\"(//dl\.smw.*?)\""
    size_re       = r"\t\t\t([0-9.]+ [KM])iB\t\t</td>"
    rating_re     = r"\t\t\t([0-9.]+)\t\t</td>"
    authors_re    = r"\t\t\t([^<].*)\t\t"
    authors_re2   = r"class=\"un\">(.*?)</a>"
    authors_re3   = r"class=\"un\">(.*?)</a></span>, ([^<].*?)</span>"
    difficulty_re = r"\t\t\t(.*?)\t\t</td>"
    exits_re      = r"\t(\d+) exit"
    featured_re   = r"\t\t\t(Yes|No)\t\t</td>"
    demo_re       = r"\t\t\t(Yes|No)\t\t</td>"
    date_re       = r"\t\t\t\t\t\t\t<span class=\"gray small\">Added: <time datetime=\"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\">.{22}</time></span>"
    idname_re     = r"id=(\d+)\">(.*?)</a>"

    date_format = DateFormat("y-m-dTH:M:S")

    for line in eachline(file)
        last_twenty = vcat(last_twenty[2:end], line)
        downloads_match = match(downloads_re, line)
        if !isnothing(downloads_match)
            linkline       = last_twenty[end - 1]
            sizeline       = last_twenty[end - 3]
            ratingline     = last_twenty[end - 5]
            authorline     = last_twenty[end - 7]
            difficultyline = last_twenty[end - 9]
            exitsline      = last_twenty[end - 11]
            featuredline   = last_twenty[end - 13]
            demoline       = last_twenty[end - 15]
            dateline       = last_twenty[end - 18]
            idnameline     = last_twenty[end - 19]
            rating_match = match(rating_re, ratingline)
            idname_match = match(idname_re, idnameline)

            push!(stats["downloads"],
                  string(parse(DownloadsType,
                               replace(downloads_match.captures[1], ',' => ""))))
            push!(stats["link"],
                  string(LinkType("https:" * match(link_re, linkline).captures[1])))
            push!(stats["size"], string(process_size(sizeline, size_re)))
            push!(stats["rating"], string(isnothing(rating_match) ? RatingType.a()
                                          : parse(RatingType.b, rating_match.captures[1])))
            push!(stats["authors"],
                  string(process_authors(authorline, authors_re, authors_re2, authors_re3)))
            push!(stats["difficulty"],
                  string(DifficultyType(match(difficulty_re, difficultyline).captures[1])))
            push!(stats["exits"],
                  string(parse(ExitsType, match(exits_re, exitsline).captures[1])))
            push!(stats["featured"], string(
                    FeaturedType(match(featured_re, featuredline).captures[1] == "Yes")))
            push!(stats["demo"],
                  string(DemoType(match(demo_re, demoline).captures[1] == "Yes")))
            push!(stats["date"],
                  string(DateType(match(date_re, dateline).captures[1], date_format)))
            push!(stats["id"], string(IdType(idname_match.captures[1])))
            push!(stats["name"],
                  string(NameType(replace(replace(idname_match.captures[2],
                                                  "&quot;" => '"'), "&amp;" => '&'))
))

        end
    end
    return stats
end

function process_size(line, re)::SizeType
    sizestring = match(re, line).captures[1]
    splitstring = split(sizestring, ' ')
    number = parse(Float32, splitstring[1])
    unit = splitstring[2][1]
    multiplier = Dict{Char, Float32}('K' => 1f3, 'M' => 1f6)[unit]
    round(SizeType, number * multiplier)
end

function process_authors(line, re, re2, re3)::AuthorsType
    m = match(re, line)
    if isnothing(m)
        m = match(re3, line)
        !isnothing(m) && return m.captures[1:2]
        res = AuthorsType()
        m = match(re2, line)
        offset::UInt = 0
        while !isnothing(m)
            push!(res, m.captures[1])
            offset += m.offset
            m = match(re2, line[offset + 1:end])
        end
        return res
    end
    return [String(m.captures[1])]
end

function writestats(file::AbstractString=hack_list,
                    outfile::AbstractString=outfile)
    stats = getstats(file)
    write(outfile, stats; delim=';')
end


if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 2 || length(ARGS) == 1 && ARGS[1] == "-h"
        print("""
              Usage: $PROGRAM_FILE [<infile>] [<outfile>]

                 infile:  The level list HTML file to read stats from (default:
                          ../stats/hack_list.html (relative to this file's directory)).
                 outfile: Which file to save the results to (default:
                          ../stats/hack_stats.csv (relative to this file's directory)).
              """)
        exit(length(ARGS) > 2 ? 1 : 0)
    end
    writestats(ARGS...)
end

end # module

