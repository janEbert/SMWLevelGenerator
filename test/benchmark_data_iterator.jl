using BenchmarkTools
using SMWLevelGenerator

db = loaddb(joinpath(@__DIR__, "..", "levels_3d_flags_t.jdb"))
iter = SMWLevelGenerator.DataIterator.dataiteratorchannel(db, 4, Val(false), Val(true))
@btime begin
    SMWLevelGenerator,DataIterator.dataiterator!($iter, $db, 1:length($db), as_matrix=true)
    for i in 1:length($db)
        take!($iter)
    end
end
close(iter)

db = loaddb(joinpath(@__DIR__, "..", "levels_2d_flags_t.jdb"))
iter = SMWLevelGenerator.DataIterator.dataiteratorchannel(db, 4, Val(false), Val(true))
@btime begin
    SMWLevelGenerator,DataIterator.dataiterator!($iter, $db, 1:length($db), as_matrix=true)
    for i in 1:length($db)
        take!($iter)
    end
end
close(iter)

