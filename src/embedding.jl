
abstract type AbstractEmbedderModel end

"""
    EmbedderModel

A model for embedding passages, including the encoder and the ONNX session for inference.

For embedding, use as `embed(embedder, passages)` or as a functor `embedder(passages)`.

"""
struct EmbedderModel <: AbstractEmbedderModel
    alias::Symbol
    encoder::BertTextEncoder
    session::ORT.InferenceSession
end
function Base.show(io::IO, result::AbstractEmbedderModel)
    dump(io, result; maxdepth = 1)
end

function EmbedderModel(alias::Symbol = :tiny)
    encoder, session = load_model(alias)
    EmbedderModel(alias, encoder, session)
end

"""
    EmbedResult{T <: Real}

The result of embedding passages.

# Fields
- `embeddings::AbstractArray{T}`: The embeddings of the passages. With property `embeddings` as column-major matrix of size `(batch_size, embedding_dimension)`.
- `elapsed::Float64`: The time taken to embed the passages.
"""
struct EmbedResult{T <: Real}
    embeddings::AbstractArray{T}
    elapsed::Float64
end
function Base.show(io::IO, result::EmbedResult)
    dump(io, result; maxdepth = 1)
end

"""
    embed(
        embedder::EmbedderModel, passages::AbstractVector{<:AbstractString})

Embeds `passages` using the given `embedder` model.

# Arguments:
- `embedder::EmbedderModel`: The embedder model to use.
- `passages::AbstractVector{<:AbstractString}`: The passages to embed.

# Returns
- `EmbedResult`: The embeddings of the passages. With property `embeddings` as column-major matrix of size `(batch_size, embedding_dimension)`.

# Example
```julia
model = EmbedderModel(:tiny_embed)
result = embed(model, ["Hello, how are you?", "How is it going?"])
result.embeddings # 312x2 matrix of Float32
```
"""
function embed(
        embedder::EmbedderModel, passages::AbstractVector{<:AbstractString})
    t = @elapsed begin
        token_ids, token_type_ids, attention_mask = encode(embedder.encoder, passages)
        ## transpose as the model expects row-major
        ## TODO: investigate pre-warming the session with padded inputs
        ## TODO: investigate performnance on materialized inputs
        onnx_input = Dict("input_ids" => token_ids', "attention_mask" => attention_mask')
        out = embedder.session(onnx_input)
        ## Permute dimensions to return column-major embeddings, ie, batch-size X embedding-size
        embeddings = out["avg_embeddings"] |> permutedims
    end
    EmbedResult(embeddings, t)
end

function embed(embedder::EmbedderModel, passages::AbstractString)
    embed(embedder, [passages])
end

function (embedder::EmbedderModel)(
        passages::Union{AbstractString, AbstractVector{<:AbstractString}}; kwargs...)
    embed(embedder, passages; kwargs...)
end
