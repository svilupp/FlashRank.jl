
abstract type AbstractRankerModel end

struct RankerModel <: AbstractRankerModel
    model_type::Symbol
    encoder::Any
    onnx_model::Any
end

function RankerModel(model_type::Symbol)
    encoder, onnx_model = load_model(model_type)
    new(model_type, encoder, onnx_model)
end

struct RankResult{T <: AbstractString}
    query::String
    docs::Vector{T}
    positions::Vector{Int}
    scores::Vector{Float32}
    elapsed::Float64
end

using LinearAlgebra: exp
using Statistics: mean

function rank(ranker::RankerModel, query::String, docs::Vector{Dict})
    t = @elapsed begin
        passages = [doc["text"] for doc in docs]
        token_ids, token_type_ids, attention_mask = encode(ranker.encoder, query, passages)
        onnx_input = Dict("input_ids" => token_ids', "token_type_ids" => token_type_ids',
            "attention_mask" => attention_mask')
        out = ranker.onnx_model(onnx_input)
    end
    # Sort and prepare results
    logits = out["logits"]
    @assert size(logits, 2)==1 "Logits are not binary, more than one class detected"
    probas = @. 1 / (1 + exp(-vec(logits)))
    sorted_indices = sortperm(probas, rev = true)
    sorted_passages = passages[sorted_indices]
    RankResult(
        query, sorted_passages, sorted_indices, scores[sorted_indices] .|> Float32, t)
end
