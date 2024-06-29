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
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false,
        max_tokens::Union{Nothing, Int} = enc.trunc)

Tokenizes the text and returns the tokens or token IDs (to skip looking up the IDs twice).

# Arguments
- `add_special_tokens::Bool = true`: Add special tokens at the beginning and end of the text.
- `add_end_token::Bool = true`: Add end token at the end of the text.
- `token_ids::Bool = false`: If true, return the token IDs directly. Otherwise, return the tokens.
- `max_tokens::Union{Nothing, Int} = enc.trunc`: The maximum number of tokens to return (usually defined by the model).
"""
function tokenize(enc::BertTextEncoder, text::AbstractString;
        add_special_tokens::Bool = true, add_end_token::Bool = true, token_ids::Bool = false,
        max_tokens::Union{Nothing, Int} = enc.trunc)
    tokens = token_ids ? Int[] : String[]
    if add_special_tokens
        token = token_ids ? enc.vocab[enc.startsym] : enc.startsym
        push!(tokens, token)
    end
    for token in bert_uncased_tokenizer(text)
        append!(tokens, enc.wp(token; token_ids))
    end
    if !isnothing(max_tokens) && length(tokens) > (max_tokens - 1)
        tokens = tokens[1:(max_tokens - 1)]
    end
    if add_special_tokens || add_end_token
        token = token_ids ? enc.vocab[enc.endsym] : enc.endsym
        push!(tokens, token)
    end
    return tokens
end

"""
    encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true,
        max_tokens::Int = enc.trunc, split_instead_trunc::Bool = false)

Encodes the text and returns the token IDs, token type IDs, and attention mask.

We enforce `max_tokens` to be a concrete number here to be able to do `split_instead_trunc`.
`split_instead_trunc` splits any long sequences into several smaller ones.
"""
function encode(enc::BertTextEncoder, text::String; add_special_tokens::Bool = true,
        max_tokens::Int = enc.trunc, split_instead_trunc::Bool = false)
    if !split_instead_trunc
        ## Standard run - if text is longer, we truncate it and ignore
        token_ids = tokenize(enc, text; add_special_tokens, token_ids = true, max_tokens)
        # Zero indexed as models are trained for Python
        token_type_ids = zeros(Int, length(token_ids))
        attention_mask = ones(Int, length(token_ids))
    else
        ## Split run - if text is longer, we split it into multiple chunks and encode them separately
        ## Only possible with a single string to know where the chunks belong to
        ## tokenize without special tokens at first
        token_ids = tokenize(enc, text; add_special_tokens = false,
            token_ids = true, max_tokens = nothing)
        ## determine correct chunk size
        start_token = enc.vocab[enc.startsym]
        end_token = enc.vocab[enc.endsym]
        chunk_size = max_tokens - 2 * add_special_tokens
        itr = Iterators.partition(token_ids, chunk_size)
        num_chunks = length(itr)
        ## split vector in several
        mat_token_ids = zeros(Int, max_tokens, num_chunks)
        token_type_ids = zeros(Int, max_tokens, num_chunks)
        attention_mask = zeros(Int, max_tokens, num_chunks)
        @inbounds for (i, chunk) in enumerate(itr)
            if add_special_tokens
                mat_token_ids[1, i] = start_token
                attention_mask[1, i] = 1
            end
            for ri in eachindex(chunk)
                ## if special token, we shift all items by 1 down
                row_idx = add_special_tokens ? ri + 1 : ri
                mat_token_ids[row_idx, i] = chunk[ri]
                attention_mask[row_idx, i] = 1
            end
            if add_special_tokens
                row_idx = 2 + length(chunk)
                mat_token_ids[row_idx, i] = end_token
                attention_mask[row_idx, i] = 1
            end
        end
        token_ids = mat_token_ids
    end
    return token_ids, token_type_ids, attention_mask
end

function encode(enc::BertTextEncoder, query::AbstractString,
        passage::AbstractString; add_special_tokens::Bool = true)
    ## Tokenize texts
    token_ids1 = tokenize(enc, query; add_special_tokens, token_ids = true)
    token_ids2 = tokenize(enc, passage; add_special_tokens = false,
        add_end_token = add_special_tokens, token_ids = true)
    token_type_ids = vcat(zeros(Int, length(token_ids1)), ones(Int, length(token_ids2)))
    token_ids = vcat(token_ids1, token_ids2)

    ## check if we exceed truncation
    if !isnothing(enc.trunc) && (length(token_ids)) > enc.trunc
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
           x -> isnothing(enc.trunc) ? x : min(x, enc.trunc)

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

# For multiple documents
function FlashRank.encode(enc::BertTextEncoder, passages::AbstractVector{<:AbstractString};
        add_special_tokens::Bool = true)
    tokens_vec = [tokenize(enc, passage; add_special_tokens = true, token_ids = true)
                  for passage in passages]
    max_len = maximum(length, tokens_vec) |>
              x -> isnothing(enc.trunc) ? x : min(x, enc.trunc)

    ## Assumes that padding is done with token ID 0
    token_ids = zeros(Int, max_len, length(passages))
    # Zero indexed as models are trained for Python
    token_type_ids = zeros(Int, max_len, length(passages))
    attention_mask = zeros(Int, max_len, length(passages))

    ## Encode to token IDS
    @inbounds for j in eachindex(tokens_vec)
        tokens = tokens_vec[j]
        for i in eachindex(tokens)
            if i > max_len
                break
            elseif i == max_len
                ## give [SEP] token
                token_ids[i, j] = enc.vocab[enc.endsym]
            else
                ## fill the tokens
                token_ids[i, j] = tokens[i]
            end
            attention_mask[i, j] = 1
        end
    end
    return token_ids, token_type_ids, attention_mask
end