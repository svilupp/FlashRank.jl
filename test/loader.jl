using FlashRank: load_model, BertTextEncoder, ORT

@testset "load_model" begin
    enc, sess = load_model(:tiny)
    @test enc isa BertTextEncoder
    @test sess isa ORT.InferenceSession
    @test_throws ArgumentError load_model(:notexistent)
end