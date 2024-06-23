using FlashRank: RankerModel, rank, RankResult

@testset "rank" begin
    ranker = RankerModel(:tiny)

    query = "How do you like New York?"
    passages = [
        "The archaeopteryx fossil exhibited characteristics of both birds and dinosaurs.",
        "El niño jugaba en el jardín mientras su madre preparaba la cena.",
        "Y'all better not miss this lit party tonight, it's gonna be epic!",
        "The antiestablishmentarianism sentiment was growing among the populace.",
        "I ❤️ New York! It's the best city ever!!! #NYC #Love",
        "The qwuick brown fox jumps ovr the lazy dog.",
        "Je suis très happy to see you today, my friend.",
        "The bass was so loud it shook the entire house.",
        "iPhone users often prefer their devices over Android phones.",
        "It's well-known that time-saving techniques are essential for efficiency."
    ]
    result = rank(ranker, query, passages)
    @test result.query == query
    @test passages[result.positions] == result.docs
    @test result.docs[1] == passages[5]

    logits = [-11.545736, -11.568807, -11.534985, -11.548196, -7.750709,
        -11.548166, -11.535543, -11.563152, -11.579304, -11.526732]
    probas = 1 ./ (1 .+ exp.(-vec(logits)))
    @test isapprox(probas[result.positions], result.scores; atol = 5e-4)

    ## Truncate results to top 5
    result = rank(ranker, query, passages; top_n = 5)
    @test result.query == query
    @test passages[result.positions] == result.docs
    @test result.docs[1] == passages[5]
    @test length(result.docs) == 5
    @test isapprox(result.scores[1], 0.0004; atol = 5e-4)
end

@testset "show-ranker" begin
    ranker = RankerModel(:tiny)
    io = IOBuffer()
    show(io, ranker)
    output = String(take!(io))
    @test occursin("RankerModel", output)
    @test occursin("tiny", output)
    @test occursin("BertTextEncoder", output)
    @test occursin("InferenceSession", output)

    result = RankResult("a", ["b", "c"], [1, 2], [0.1f0, 0.2f0], 0.3)
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("RankResult", output)
    @test occursin("a", output)
    @test occursin("positions", output)
    @test occursin("scores", output)
    @test occursin("elapsed", output)
end