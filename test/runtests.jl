using FlashRank
using Test
using Aqua

@testset "FlashRank.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        ## skip ambiguities due to DAT and StringViews fails
        Aqua.test_all(FlashRank; ambiguities = false)
    end
    include("loader.jl")
    include("encoder.jl")
    include("ranking.jl")
    include("embedding.jl")
end
