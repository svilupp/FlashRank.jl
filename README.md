# FlashRank.jl 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://svilupp.github.io/FlashRank.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://svilupp.github.io/FlashRank.jl/dev/) [![Build Status](https://github.com/svilupp/FlashRank.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/svilupp/FlashRank.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/svilupp/FlashRank.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/svilupp/FlashRank.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

FlashRank.jl is inspired by the awesome [FlashRank Python package](https://github.com/PrithivirajDamodaran/FlashRank), originally developed by Prithiviraj Damodaran. This package leverages model weights from [Prithiviraj's HF repo](https://huggingface.co/prithivida/flashrank) and [Svilupp's HF repo](https://huggingface.co/svilupp/onnx-cross-encoders) to provide **a fast and efficient way to rank documents relevant to any given query without GPUs and large dependencies**. 

This enhances Retrieval Augmented Generation (RAG) pipelines by prioritizing the most suitable documents. The smallest model can be run on almost any machine.

## Features
- Four ranking models:
  - **Tiny (~4MB, INT8):** [ms-marco-TinyBERT-L-2-v2 (default)](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2) (alias `:tiny`)
  - **MiniLM L-4 (~70MB, FP32):** [ms-marco-MiniLM-L-4-v2 ONNX](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-4-v2) (alias `:mini4`)
  - **MiniLM L-6 (~83.4MB, FP32):** [ms-marco-MiniLM-L-6-v2 ONNX](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (alias `:mini6`)
  - **MiniLM L-12 (~23MB, INT8):** [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) (alias `:mini` or `mini12`)
- Lightweight dependencies, avoiding heavy frameworks like Flux and CUDA for ease of integration.

How fast is it? 
With the Tiny model, you can rank 100 documents in ~0.1 seconds on a laptop. With the MiniLM (12 layers) model, you can rank 100 documents in ~0.4 seconds.

Tip: Pick the largest model that you can afford with your latency budget, ie, MiniLM L-12 is the slowest but has the best accuracy.

Note that we're using BERT models with a maximum chunk size of 512 tokens (anything over will be truncated).

## Installation
Add it to your environment simply with:

```julia
using Pkg
Pkg.activate(".")
Pkg.add("FlashRank")
```

## Usage
Ranking your documents for a given query is as simple as:

```julia
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
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

`result` is of type `RankResult` and contains the sorted passages, their scores (0-1, where 1 is the best) and the positions of the sorted documents (referring to the original `passages` vector).

Here's a brief outline of how you can integrate FlashRank.jl into your [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) RAG pipeline.

For a full example, see `examples/prompting_tools_integration.jl`.

```julia
using FlashRank
using PromptingTools
using PromptingTools.Experimental.RAGTools
const RT = PromptingTools.Experimental.RAGTools

# Wrap the model to be a valid Ranker recognized by RAGTools
# It will be provided to the airag/rerank function to avoid instantiating it on every call
struct FlashRanker <: RT.AbstractReranker
    model::RankerModel
end
reranker = RankerModel(:tiny) |> FlashRanker

# Define the method for ranking with it
function RT.rerank(
        reranker::FlashRanker, index::RT.AbstractDocumentIndex, question::AbstractString,
        candidates::RT.AbstractCandidateChunks; kwargs...)
    ## omitted for brevity
    ## See examples/prompting_tools_integration.jl for details
end

## Apply to the pipeline configuration, eg, 
cfg = RAGConfig(; retriever=RT.AdvancedRetriever(; reranker))
## assumes existing index
question = "Tell me about prehistoric animals"
result = airag(cfg, index; question, return_all = true)
```

## Advanced Usage

You can also leverage quite "coarse" but fast embeddings with the `tiny_embed` model (Bert-L4).

```julia
embedder = FlashRank.EmbedderModel(:tiny_embed)

passages = ["This is a test", "This is another test"]
result = FlashRank.embed(embedder, passages)
```

## Acknowledgments
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) and [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) have been essential in the development of this package.
- Special thanks to Prithiviraj Damodaran for the original FlashRank and the INT8 quantized model weights.
- And to Transformers.jl for the WordPiece implementation and BERT tokenizer which have been forked for this package (to minimize dependencies).

## Roadmap
- [ ] Provide package extension for PromptingTools
- [ ] Bring even smaller models (eg, Ber-L2-128D)
- [ ] Introduce a simply length-based adjustment to embedding similarity score
- [ ] Re-upload embed models with mask-based pooling (no real difference, just theoretically correct)