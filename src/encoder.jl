"""
    BertTextEncoder

The text encoder for Bert model (WordPiece tokenization).
"""
@kwdef struct BertTextEncoder
    wp::WordPiece
    vocab::Dict{String, Int}
    startsym::String = "[CLS]"
    endsym::String = "[SEP]"
    padsym::String = "[PAD]"
    trunc::Union{Nothing, Int} = nothing
end
function Base.show(io::IO, enc::BertTextEncoder)
    dump(io, enc; maxdepth = 1)
end

function BertTextEncoder(
        wp, vocab; startsym = "[CLS]", endsym = "[SEP]", padsym = "[PAD]", trunc = nothing)
    haskey(vocab, startsym) ||
        @warn "startsym $startsym not in vocabulary, this might cause problem."
    haskey(vocab, endsym) ||
        @warn "endsym $endsym not in vocabulary, this might cause problem."
    haskey(vocab, padsym) ||
        @warn "padsym $padsym not in vocabulary, this might cause problem."
    return BertTextEncoder(wp, vocab, startsym, endsym, padsym, trunc)
end

function tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true)
    tokens = String[]
    if add_special_tokens
        push!(tokens, enc.startsym)
    end
    for token in bert_uncased_tokenizer(text)
        append!(tokens, enc.wp(token))
    end
    if add_special_tokens || add_end_token
        push!(tokens, enc.endsym)
    end
    return tokens
end

function encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true)
    tokens = tokenize(enc, text; add_special_tokens)

    unki = enc.vocab[DAT.decode(wp.trie, wp.unki)]
    token_ids = [get(enc.vocab, t, unki) for t in tokens]
    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, length(tokens))
    attention_mask = ones(Int, length(tokens))
    return token_ids, token_type_ids, attention_mask
end

function encode(enc::BertTextEncoder, query::AbstractString,
        passage::AbstractString; add_special_tokens::Bool = true)
    ## Tokenize texts
    tokens1 = tokenize(enc, query; add_special_tokens)
    tokens2 = tokenize(enc, passage; add_special_tokens = false,
        add_end_token = add_special_tokens)
    token_type_ids = vcat(zeros(Int, length(tokens1)), ones(Int, length(tokens2)))

    ## Encode to token IDS
    unki = enc.vocab[DAT.decode(wp.trie, wp.unki)]
    token_ids = [get(enc.vocab, t, unki) for t in tokens1]
    append!(token_ids, [get(enc.vocab, t, unki) for t in tokens2])
    # Zero indexed as models are trained for Python
    attention_mask = ones(Int, length(token_ids))
    return token_ids, token_type_ids, attention_mask
end

function encode(enc::BertTextEncoder, query::AbstractString,
        passages::Vector{<:AbstractString}; add_special_tokens::Bool = true)

    ## tokenize query, it will be repeated
    tokens1 = tokenize(enc, query; add_special_tokens)

    tokens2_vec = [tokenize(enc, passage; add_special_tokens = false,
                       add_end_token = add_special_tokens) for passage in passages]
    len_ = maximum(length, tokens2_vec) + length(tokens1)

    token_ids = zeros(Int, len_, length(passages))
    token_type_ids = zeros(Int, len_, length(passages))
    attention_mask = zeros(Int, len_, length(passages))

    ## Encode to token IDS
    unki = enc.vocab[DAT.decode(wp.trie, wp.unki)]
    tokens1_ids = [get(enc.vocab, t, unki) for t in tokens1]
    tokens1_len = length(tokens1_ids)
    @inbounds for j in eachindex(tokens2_vec)
        token_ids[1:tokens1_len, j] .= tokens1_ids
        attention_mask[1:tokens1_len, j] .= 1

        tokens2 = tokens2_vec[j]
        for i in eachindex(tokens2)
            token_ids[tokens1_len + i, j] = get(enc.vocab, tokens2[i], unki)
            token_type_ids[tokens1_len + i, j] = 1
            attention_mask[tokens1_len + i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end
