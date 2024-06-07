
abstract type AbstractRankerModel end

"""
    RankerModel

A model for ranking passages, including the encoder and the ONNX session for inference.

For ranking, use as `rank(ranker, query, passages)` or as a functor `ranker(query, passages)`.

"""
struct RankerModel <: AbstractRankerModel
    alias::Symbol
    encoder::BertTextEncoder
    session::ORT.InferenceSession
end
function Base.show(io::IO, result::AbstractRankerModel)
    dump(io, result; maxdepth = 1)
end

function RankerModel(alias::Symbol = :tiny)
    encoder, session = load_model(alias)
    RankerModel(alias, encoder, session)
end

struct RankResult{T <: AbstractString}
    query::String
    docs::Vector{T}
    positions::Vector{Int}
    scores::Vector{Float32}
    elapsed::Float64
end
function Base.show(io::IO, result::RankResult)
    dump(io, result; maxdepth = 1)
end

"""
    rank(
        ranker::RankerModel, query::AbstractString, passages::AbstractVector{<:AbstractString};
        top_n = length(passages))

Ranks `passages` for a given `query` using the given `ranker` model. Ranking should determine higher suitability to provide an answer to the query (higher score is better).

# Arguments:
- `ranker::RankerModel`: The ranker model to use.
- `query::AbstractString`: The query to rank passages for.
- `passages::AbstractVector{<:AbstractString}`: The passages to rank.
- `top_n`: The number of most relevant documents to return. Default is `length(passages)`.
"""
function rank(
        ranker::RankerModel, query::AbstractString, passages::AbstractVector{<:AbstractString};
        top_n = length(passages))
    t = @elapsed begin
        token_ids, token_type_ids, attention_mask = encode(ranker.encoder, query, passages)
        ## transpose as the model expects row-major
        ## TODO: investigate pre-warming the session with padded inputs
        ## TODO: investigate performnance on materialized inputs
        onnx_input = Dict("input_ids" => token_ids', "token_type_ids" => token_type_ids',
            "attention_mask" => attention_mask')
        out = ranker.session(onnx_input)
    end
    # Sort and prepare results
    logits = out["logits"]
    @assert size(logits, 2)==1 "Logits are not binary, more than one class detected"
    probas = 1 ./ (1 .+ exp.(-vec(logits)))
    sorted_indices = sortperm(probas, rev = true) |> x -> first(x, top_n)
    sorted_passages = passages[sorted_indices]
    RankResult(
        query, sorted_passages, sorted_indices, @view(probas[sorted_indices]) .|> Float32, t)
end

function (ranker::RankerModel)(
        query::AbstractString, passages::AbstractVector{<:AbstractString}; kwargs...)
    rank(ranker, query, passages; kwargs...)
end
