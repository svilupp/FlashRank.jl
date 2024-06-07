using FlashRank: RankerModel, rank

@testset "rank" begin
    ranker = RankerModel(:tiny)

    query = "Tell me about prehistoric animals"
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
    @test result.docs[1] == passages[1]

    logits = [-10.512585, -11.525919, -11.541619, -11.473759, -11.555203,
        -11.479718, -11.492066, -11.542512, -11.570846, -11.539953]
    probas = 1 ./ (1 .+ exp.(-vec(logits)))
    @test isapprox(probas[result.positions], result.scores)

    ## Truncate results to top 5
    result = rank(ranker, query, passages; top_n = 5)
    @test result.query == query
    @test passages[result.positions] == result.docs
    @test result.docs[1] == passages[1]
    @test length(result.docs) == 5
    @test result.scores[1] ≈ 2.7191343f-5
end