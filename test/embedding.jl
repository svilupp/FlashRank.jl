using FlashRank: EmbedderModel, embed, EmbedResult

@testset "embed" begin
    embedder = EmbedderModel(:tiny_embed)

    texts = ["Hello, how are you?", "How is it going?",
        "I am fine, thank you.", "I had a coffee for breakfast"]

    # Calculate cosine similarity between the embeddings of the three texts
    function cosine_similarity(x, y)
        dot_product = sum(x .* y)
        norm_x = sqrt(sum(x .^ 2))
        norm_y = sqrt(sum(y .^ 2))
        return dot_product / (norm_x * norm_y)
    end

    result = embed(embedder, texts)
    @test result.embeddings isa AbstractArray{Float32}
    @test size(result.embeddings) == (312, 4)
    embeddings = result.embeddings
    cos_sim_12 = cosine_similarity(embeddings[:, 1], embeddings[:, 2])
    cos_sim_13 = cosine_similarity(embeddings[:, 1], embeddings[:, 3])
    cos_sim_14 = cosine_similarity(embeddings[:, 1], embeddings[:, 4])
    cos_sim_23 = cosine_similarity(embeddings[:, 2], embeddings[:, 3])
    cos_sim_24 = cosine_similarity(embeddings[:, 2], embeddings[:, 4])
    cos_sim_34 = cosine_similarity(embeddings[:, 3], embeddings[:, 4])

    @test isapprox(cos_sim_12, 0.75847805; atol = 5e-2)
    @test isapprox(cos_sim_13, 0.4089551; atol = 5e-2)
    @test isapprox(cos_sim_14, 0.28373927; atol = 5e-2)
    @test isapprox(cos_sim_23, 0.32157198; atol = 5e-2)
    @test isapprox(cos_sim_24, 0.35012797; atol = 5e-2)
    @test isapprox(cos_sim_34, 0.27234405; atol = 5e-2)

    ## Different functor interface
    result = embedder(texts)
    @test result.embeddings isa AbstractArray{Float32}
    @test size(result.embeddings) == (312, 4)

    result = embedder(texts[1])
    @test result.embeddings isa AbstractArray{Float32}
    @test size(result.embeddings) == (312, 1)

    ## Splitting - no effect
    result = embed(embedder, texts[1]; split_instead_trunc = true)
    @test size(result.embeddings) == (312, 1)

    # split when long text
    long_text = repeat("Hello, how are you? ", 200)
    result = embed(embedder, long_text; split_instead_trunc = true)
    @test size(result.embeddings) == (312, 3)
end

@testset "show-embedding" begin
    embedder = EmbedderModel(:tiny_embed)
    io = IOBuffer()
    show(io, embedder)
    output = String(take!(io))
    @test occursin("tiny_embed", output)
    @test occursin("BertTextEncoder", output)
    @test occursin("InferenceSession", output)

    result = EmbedResult(rand(Float32, 312, 4), 0.1)
    io = IOBuffer()
    show(io, result)
    output = String(take!(io))
    @test occursin("EmbedResult{Float32}", output)
    @test occursin("(312, 4)", output)
    @test occursin("0.1", output)
    @test occursin("embeddings", output)
    @test occursin("elapsed", output)
end