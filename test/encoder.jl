using FlashRank: RankerModel, tokenize, encode

@testset "tokenize,encode" begin

    # Test 1: Rare words and scientific terminology
    ranker = RankerModel(:tiny)
    encoder = ranker.encoder

    s1 = "The archaeopteryx fossil exhibited characteristics of both birds and dinosaurs."
    tokens = tokenize(encoder, s1)
    expected_tokens1 = [
        "[CLS]", "the", "arch", "##ae", "##op", "##tery", "##x", "fossil", "exhibited",
        "characteristics", "of", "both", "birds", "and", "dinosaurs", ".", "[SEP]"]
    @test tokens == expected_tokens1
    expected_ids1 = [101, 1996, 7905, 6679, 7361, 20902, 2595, 10725,
        8176, 6459, 1997, 2119, 5055, 1998, 18148, 1012, 102]
    out = encode(encoder, s1)
    @test out[1] == expected_ids1
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 2: Foreign languages
    s2 = "El niño jugaba en el jardín mientras su madre preparaba la cena."
    tokens = tokenize(encoder, s2)
    expected_tokens2 = ["[CLS]", "el", "nino", "jug", "##aba", "en", "el", "jar",
        "##din", "mi", "##ent", "##ras", "su", "mad", "##re",
        "prep", "##ara", "##ba", "la", "ce", "##na", ".", "[SEP]"]
    expected_ids2 = [101, 3449, 20801, 26536, 19736, 4372, 3449, 15723, 8718, 2771, 4765,
        8180, 10514, 5506, 2890, 17463, 5400, 3676, 2474, 8292, 2532, 1012, 102]
    @test tokens == expected_tokens2
    out = encode(encoder, s2)
    @test out[1] == expected_ids2
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 3: Slang and informal language
    s3 = "Y'all better not miss this lit party tonight, it's gonna be epic!"
    tokens = tokenize(encoder, s3)
    expected_tokens = ["[CLS]", "y",
        "'", "all", "better", "not", "miss", "this", "lit", "party", "tonight", ",", "it", "'",
        "s", "gonna", "be", "epic", "!", "[SEP]"]
    expected_ids = [101, 1061, 1005, 2035, 2488, 2025, 3335, 2023, 5507, 2283,
        3892, 1010, 2009, 1005, 1055, 6069, 2022, 8680, 999, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s3)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 4: Concatenated words and compound words
    s4 = "The antiestablishmentarianism sentiment was growing among the populace."
    tokens = tokenize(encoder, s4)
    expected_tokens = [
        "[CLS]", "the", "anti", "##est", "##ab", "##lish", "##ment", "##arian", "##ism",
        "sentiment", "was", "growing", "among", "the", "populace", ".", "[SEP]"]
    expected_ids = [101, 1996, 3424, 4355, 7875, 13602, 3672, 12199, 2964,
        15792, 2001, 3652, 2426, 1996, 22508, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s4)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 5: Special characters and emojis
    s5 = "I ❤️ New York! It's the best city ever!!! #NYC #Love"
    tokens = tokenize(encoder, s5)
    expected_tokens = ["[CLS]",
        "i",
        "[UNK]",
        "new",
        "york",
        "!",
        "it",
        "'", "s", "the", "best", "city", "ever", "!",
        "!", "!", "#", "nyc", "#", "love", "[SEP]"]
    expected_ids = [101, 1045, 100, 2047, 2259, 999, 2009, 1005, 1055, 1996, 2190,
        2103, 2412, 999, 999, 999, 1001, 16392, 1001, 2293, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s5)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 6: Typos and misspellings
    s6 = "The qwuick brown fox jumps ovr the lazy dog."
    tokens = tokenize(encoder, s6)
    expected_tokens = ["[CLS]", "the", "q", "##wu", "##ick", "brown", "fox",
        "jumps", "o", "##vr", "the", "lazy", "dog", ".", "[SEP]"]
    expected_ids = [101, 1996, 1053, 16050, 6799, 2829, 4419, 14523,
        1051, 19716, 1996, 13971, 3899, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s6)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 7: Mixed languages
    s7 = "Je suis très happy to see you today, my friend."
    tokens = tokenize(encoder, s7)
    expected_tokens = ["[CLS]", "je", "sui", "##s", "tres", "happy", "to",
        "see", "you", "today", ",", "my", "friend", ".", "[SEP]"]
    expected_ids = [101, 15333, 24086, 2015, 24403, 3407, 2000,
        2156, 2017, 2651, 1010, 2026, 2767, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s7)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))
    [101, 15333, 24086, 2015, 24403, 3407, 2000,
        2156, 2017, 2651, 1010, 2026, 2767, 1012, 102]

    # Test 8: Homophones and context-based interpretation
    s8 = "The bass was so loud it shook the entire house."
    tokens = tokenize(encoder, s8)
    expected_tokens = ["[CLS]", "the", "bass", "was", "so", "loud", "it",
        "shook", "the", "entire", "house", ".", "[SEP]"]
    expected_ids = [
        101, 1996, 3321, 2001, 2061, 5189, 2009, 3184, 1996, 2972, 2160, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s8)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 9: Uncommon capitalization
    s9 = "iPhone users often prefer their devices over Android phones."
    tokens = tokenize(encoder, s9)
    expected_tokens = ["[CLS]", "iphone", "users", "often", "prefer", "their",
        "devices", "over", "android", "phones", ".", "[SEP]"]
    expected_ids = [101, 18059, 5198, 2411, 9544, 2037, 5733, 2058, 11924, 11640, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s9)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Test 10: Hyphenated words and contractions
    s10 = "It's well-known that time-saving techniques are essential for efficiency."
    tokens = tokenize(encoder, s10)
    expected_tokens = ["[CLS]", "it", "'",
        "s",
        "well",
        "-",
        "known",
        "that",
        "time",
        "-",
        "saving",
        "techniques",
        "are",
        "essential",
        "for",
        "efficiency",
        ".",
        "[SEP]"]
    expected_ids = [101, 2009, 1005, 1055, 2092, 1011, 2124, 2008, 2051,
        1011, 7494, 5461, 2024, 6827, 2005, 8122, 1012, 102]
    @test tokens == expected_tokens
    out = encode(encoder, s10)
    @test out[1] == expected_ids
    @test length(out[1]) == length(out[2]) == length(out[3])
    @test out[3] == ones(length(out[1]))

    # Extra long input (over 512 tokens)
    s11 = repeat(s8, 60)
    length(s11)
    tokens = tokenize(encoder, s11)
    # first three: ['[CLS]', 'the', 'bass',
    @test tokens[1:3] == ["[CLS]", "the", "bass"]
    # last three: 'was', 'so', '[SEP]']
    @test tokens[(end - 2):end] == ["was", "so", "[SEP]"]
    @test length(tokens) == 512

    out = encode(encoder, s11)
    @test length(out[1]) == length(out[2]) == length(out[3]) == 512

    # processing of query + passage
    out = encode(encoder, s1, s2)
    @test out[1] == vcat(expected_ids1, expected_ids2[2:end])
    @test out[2] ==
          vcat(zeros(Int, length(expected_ids1)), ones(Int, length(expected_ids2) - 1))
    @test out[3] == ones(Int, length(expected_ids1) + length(expected_ids2) - 1)

    ## Too long input
    out = encode(encoder, s1, s11)
    @test out[1][1:length(expected_ids1)] == expected_ids1
    @test length(out[1]) == length(out[2]) == length(out[3]) == 512
    @test out[1][end] == 102
    @test out[2][end] == 1
    @test out[3] == ones(Int, 512)

    # Batch process query + passages
    out = encode(encoder, s1, [s2])
    @test size(out[1]) == (length(expected_ids1) + length(expected_ids2) - 1, 1)
    @test vec(out[1]) == vcat(expected_ids1, expected_ids2[2:end])
    @test vec(out[2]) ==
          vcat(zeros(Int, length(expected_ids1)), ones(Int, length(expected_ids2) - 1))
    @test out[3] == ones(Int, length(expected_ids1) + length(expected_ids2) - 1, 1)

    # Too long input
    out = encode(encoder, s1, [s1, s11, s2])
    @test size(out[1]) == size(out[2]) == size(out[3]) == (512, 3)
    ## Only middle sequence has CLS token, the rest is padded
    @test out[1][512, :] == [0, 102, 0]
    @test out[1][1:(length(expected_ids1) + length(expected_ids2) - 1), 3] ==
          vcat(expected_ids1, expected_ids2[2:end])
    @test count(>(0), out[1][:, 1]) == 2 * length(expected_ids1) - 1
    @test out[2][33, :] == [1, 1, 1]
    @test out[2][39, :] == [0, 1, 1]
    @test out[2][40, :] == [0, 1, 0]
    @test out[2][512, :] == [0, 1, 0]
    ## Attention mask
    @test out[3][512, :] == [0, 1, 0]
    @test sum(out[3]) == 512 + 33 + 39

    ### Encoding multiple sequences
    texts = ["Hello, how are you?", "I am fine, thank you."]
    output = encode(encoder, texts)
    @test output[1] ==
          [101 101; 7592 1045; 1010 2572; 2129 2986; 2024 1010; 2017 4067; 1029 2017;
           102 1012; 0 102]
    @test all(iszero, output[2])
    @test output[3] == [1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 1 1; 0 1]
end