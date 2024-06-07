# FlashRank.jl 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://svilupp.github.io/FlashRank.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://svilupp.github.io/FlashRank.jl/dev/) [![Build Status](https://github.com/svilupp/FlashRank.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/svilupp/FlashRank.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/svilupp/FlashRank.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/svilupp/FlashRank.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

FlashRank.jl is inspired by the awesome [FlashRank Python package](https://github.com/PrithivirajDamodaran/FlashRank), originally developed by Prithiviraj Damodaran. This package leverages model weights from [Prithiviraj's repository on Hugging Face](https://huggingface.co/prithivida/flashrank) and provides a fast and efficient way to rank documents relevant to any given query without GPUs and large dependencies. This enhances Retrieval Augmented Generation (RAG) pipelines by prioritizing the most suitable documents. The smallest model can be run on almost any machine.

## Features
- Two ranking models:
  - **Tiny (~4MB):** [ms-marco-TinyBERT-L-2-v2 (default)](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2)
  - **Mini (~23MB):** [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
- Lightweight dependencies, avoiding heavy frameworks like Flux and CUDA for ease of integration.

## Installation
To install FlashRank.jl, simply add this repository (package is not yet registered).

```julia
using Pkg
Pkg.add("https://github.com/svilupp/FlashRank.jl")
```

## Usage
Ranking your documents for a given query is as simple as:

```julia
using FlashRank

ranker = RankerModel() # Defaults to model = `:tiny`

query = "How to speedup LLMs?"
passages = [
        "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
        "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more \$\$\$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
        "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
        "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
        "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
];


result = rank(ranker, query, passages)
```

Here's how you can integrate FlashRank.jl into your [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) RAG pipeline:

```julia
using FlashRank
using PromptingTools
using PromptingTools.Experimental.RAGTools
const RT = PromptingTools.Experimental.RAGTools


struct FlashRanker <: RT.AbstractReranker
    model::RankerModel
end

# Wrap the model to be a valid Ranker recognized by RAGTools
# It will be provided to the airag/rerank function to avoid instantiating it on every call
reranker = RankerModel(:tiny) |> FlashRanker

# Define the method for ranking with it
function RT.rerank(
        reranker::FlashRanker, index::AbstractDocumentIndex, question::AbstractString,
        candidates::AbstractCandidateChunks;
        verbose::Bool = false,
        top_n::Integer = length(candidates.scores),
        return_documents::Bool = false,
        cost_tracker = Threads.Atomic{Float64}(0.0),
        kwargs...)
    # TODO: finish
end

## Apply to the pipeline configuration, eg, 
cfg = RAGConfig(; retriever=AdvancedRetriever(; reranker))
## assumes existing index
question = "Tell me about prehistoric animals"
result = airag(cfg, index; question, return_all = true)
```

## Acknowledgments
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) and [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) have been essential in the development of this package.
- Special thanks to Prithiviraj Damodaran for the original FlashRank and model weights.
- And to Transformers.jl for the WordPiece implementation and BERT tokenizer which have been forked for this package (to minimize dependencies).

## Roadmap
- [ ] Provide package extension for PromptingTools
- [ ] Extend support for more models