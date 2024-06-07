## This code is forked from Transformers.jl: https://github.com/chengchingwen/Transformers.jl
"""
    WordPiece

WordPiece is a tokenizer that splits a string into a sequence of KNOWN sub-word tokens (or token IDs).
It uses a double array trie to store the vocabulary and the index of the vocabulary.

Implementation is based on: https://github.com/chengchingwen/Transformers.jl

# Fields
- `trie::DoubleArrayTrie`: The double array trie of the vocabulary (for fast lookups of tokens).
- `index::Vector{Int}`: The index of the vocabulary. It is 0-based as we provide token IDs to models trained in python.
- `unki::Int`: The index of the unknown token in the TRIE (ie, this is not the token ID, but the trie index).
- `max_char::Int`: The maximum number of characters in a token. Default is 200.
- `subword_prefix::String`: The prefix of a sub-word token. Default is "##".
"""
struct WordPiece
    trie::DoubleArrayTrie
    ## index::DAT.CVector # changed due to bugs
    ## Returns 0-based indexing as we provide it to models trained in python
    index::Vector{Int}
    unki::Int
    max_char::Int
    subword_prefix::String
end

function WordPiece(vocab_list::Vector{String}, unk::String = "[UNK]";
        max_char = 200, subword_prefix = "##")
    trie = DoubleArrayTrie(copy(vocab_list))
    unki = DAT.lookup(trie, unk)
    index = Vector{Int}(undef, length(vocab_list))
    for (i, str) in enumerate(vocab_list)
        index[DAT.lookup(trie, str)] = i - 1
    end
    return WordPiece(trie, index, unki, max_char, subword_prefix)
end
function WordPiece(vocab::Vector{String}, unki::Int; max_char = 200, subword_prefix = "##")
    @assert 0 < unki <= length(vocab)
    unk = vocab[unki]
    return WordPiece(vocab, unk; max_char, subword_prefix)
end

struct _WithPrefix{A1, A2} <: AbstractVector{UInt8}
    x::A1
    y::A2
    offset::Int
    length::Int
end
_WithPrefix(x, y) = (lenx = length(x); _WithPrefix(x, y, lenx, lenx + length(y)))
function Base.getindex(x::_WithPrefix, i)
    offset = x.offset
    return i > offset ? x.y[i - offset] : x.x[i]
end
Base.length(x::_WithPrefix) = x.length
Base.size(x::_WithPrefix) = (length(x),)

"""
    (wp::WordPiece; token_ids::Bool = false)(x)

WordPiece functor that tokenizes a string into a sequence of tokens (or token IDs).

# Arguments
- `token_ids::Bool = false`: If true, return the token IDs directly. Otherwise, return the tokens.
"""
function (wp::WordPiece)(x::AbstractString; token_ids::Bool = false)
    result = token_ids ? Vector{Int}() : Vector{String}()
    isempty(x) && return result
    len = ncodeunits(x)
    failed = true
    s = 1
    if length(x) <= wp.max_char
        codes = codeunits(x)
        prefix = codeunits(wp.subword_prefix)
        while s <= len
            e = lastindex(x)
            failed = true
            while s <= e
                cbuf = @view codes[s:(nextind(x, e) - 1)]
                buf = isone(s) ? cbuf : _WithPrefix(prefix, cbuf)
                id = DAT.lookup(wp.trie, buf)
                if iszero(id)
                    e = prevind(x, e)
                else
                    if token_ids
                        @inbounds push!(result, wp.index[id])
                    else
                        push!(result, DAT.decode(wp.trie, id))
                    end
                    failed = false
                    s = nextind(x, e)
                    break
                end
            end
            failed && break
        end
    end

    if failed
        empty!(result)
        if token_ids
            @inbounds push!(result, wp.index[wp.unki])
        else
            push!(result, DAT.decode(wp.trie, wp.unki))
        end
    end
    return result
end

function Base.show(io::IO, wp::WordPiece)
    print(io, "WordPiece(vocab_size = ", length(wp.trie),
        ", unk = ", DAT.decode(wp.trie, wp.unki),
        ", max_char = ", wp.max_char, ')')
end