using FlashRank
using Test
using Aqua

@testset "FlashRank.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(FlashRank)
    end
    include("loader.jl")
    include("encoder.jl")
    include("ranking.jl")
end
