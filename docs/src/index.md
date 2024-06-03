```@meta
CurrentModule = FlashRank
```

# FlashRank

## Installation
To install FlashRank.jl, simply add this repository (package is not yet registered).

```julia
using Pkg
Pkg.add("https://github.com/svilupp/FlashRank.jl")
```

## Usage
Here's how you can integrate FlashRank.jl into your [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) RAG pipeline:

```julia
using FlashRank

ranker = RankerModel() # Defaults to model = `:tiny`

query = "How to speedup LLMs?"
passages = [
    Dict(
        "id" => 1,
        "text" => "Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
        "meta" => Dict("additional" => "info1")
    ),
    Dict(
        "id" => 2,
        "text" => "LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more \$\$\$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
        "meta" => Dict("additional" => "info2")
    ),
    Dict(
        "id" => 3,
        "text" => "There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
        "meta" => Dict("additional" => "info3")),
    Dict(
        "id" => 4,
        "text" => "Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
        "meta" => Dict("additional" => "info4")
    ),
    Dict(
        "id" => 5,
        "text" => "vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
        "meta" => Dict("additional" => "info5")
    )
];


result = rank(ranker, query, passages)
```

## Acknowledgments
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) and [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) have been essential in the development of this package.
- Special thanks to Prithiviraj Damodaran for the original FlashRank and model weights.
- Transformers.jl for the WordPiece implementation and BERT tokenizer which have been forked for this package.