"""
    BertTextEncoder

The text encoder for Bert model (WordPiece tokenization).

# Fields
- `wp::WordPiece`: The WordPiece tokenizer.
- `vocab::Dict{String, Int}`: The vocabulary, 0-based indexing of tokens to match Python implementation.
- `startsym::String`: The start symbol.
- `endsym::String`: The end symbol.
- `padsym::String`: The pad symbol.
- `trunc::Union{Nothing, Int}`: The truncation length. Defaults to 512 tokens.
"""
@kwdef struct BertTextEncoder
    wp::WordPiece
    vocab::Dict{String, Int}
    startsym::String = "[CLS]"
    endsym::String = "[SEP]"
    padsym::String = "[PAD]"
    trunc::Union{Nothing, Int} = 512
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

"""
    tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false)

Tokenizes the text and returns the tokens or token IDs (to skip looking up the IDs twice).

# Arguments
- `add_special_tokens::Bool = true`: Add special tokens at the beginning and end of the text.
- `add_end_token::Bool = true`: Add end token at the end of the text.
- `token_ids::Bool = false`: If true, return the token IDs directly. Otherwise, return the tokens.
"""
function tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false)
    tokens = token_ids ? Int[] : String[]
    if add_special_tokens
        token = token_ids ? enc.vocab[enc.startsym] : enc.startsym
        push!(tokens, token)
    end
    for token in bert_uncased_tokenizer(text)
        append!(tokens, enc.wp(token; token_ids))
    end
    if !isnothing(enc.trunc) && length(tokens) > (enc.trunc - 1)
        tokens = tokens[1:(enc.trunc - 1)]
    end
    if add_special_tokens || add_end_token
        token = token_ids ? enc.vocab[enc.endsym] : enc.endsym
        push!(tokens, token)
    end
    return tokens
end

function encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true)
    token_ids = tokenize(enc, text; add_special_tokens, token_ids = true)
    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, length(token_ids))
    attention_mask = ones(Int, length(token_ids))
    return token_ids, token_type_ids, attention_mask
end

function encode(enc::BertTextEncoder, query::AbstractString,
        passage::AbstractString; add_special_tokens::Bool = true)
    ## Tokenize texts
    token_ids = tokenize(enc, query; add_special_tokens, token_ids = true)
    token_ids2 = tokenize(enc, passage; add_special_tokens = false,
        add_end_token = add_special_tokens, token_ids = true)
    token_type_ids = vcat(zeros(Int, length(token_ids)), ones(Int, length(token_ids2)))

    ## check if we exceed truncation
    if !isnothing(enc.trunc) && (length(token_ids) + length(token_ids2)) > enc.trunc
        token_ids = first(token_ids, enc.trunc)
        ## add [SEP] token
        token_ids[end] = enc.vocab[enc.endsym]
        token_type_ids = first(token_type_ids, enc.trunc)
    end

    # Zero indexed as models are trained for Python
    attention_mask = ones(Int, length(token_ids))
    return token_ids, token_type_ids, attention_mask
end

function encode(enc::BertTextEncoder, query::AbstractString,
        passages::AbstractVector{<:AbstractString}; add_special_tokens::Bool = true)

    ## tokenize query, it will be repeated
    token_ids1 = tokenize(enc, query; add_special_tokens, token_ids = true)

    tokens_ids2_vec = [tokenize(enc, passage; add_special_tokens = false,
                           add_end_token = add_special_tokens, token_ids = true)
                       for passage in passages]
    len_ = maximum(length, tokens_ids2_vec) + length(token_ids1) |>
           x -> isnothing(enc.trunc) ? x : max(x, enc.trunc)

    ## Assumes that padding is done with token ID 0
    token_ids = zeros(Int, len_, length(passages))
    token_type_ids = zeros(Int, len_, length(passages))
    attention_mask = zeros(Int, len_, length(passages))

    ## Encode to token IDS
    token_ids1_len = length(token_ids1)
    @inbounds for j in eachindex(tokens_ids2_vec)
        token_ids[1:token_ids1_len, j] .= token_ids1
        attention_mask[1:token_ids1_len, j] .= 1

        tokens_ids2 = tokens_ids2_vec[j]
        for i in eachindex(tokens_ids2)
            if token_ids1_len + i > len_
                break
            elseif token_ids1_len + i == len_
                ## give [SEP] token
                token_ids[token_ids1_len + i, j] = enc.vocab[enc.endsym]
            else
                ## fill the tokens
                token_ids[token_ids1_len + i, j] = tokens_ids2[i]
            end
            token_type_ids[token_ids1_len + i, j] = 1
            attention_mask[token_ids1_len + i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end
