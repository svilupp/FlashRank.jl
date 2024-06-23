module FlashRank

using JSON3
import ONNXRunTime
const ORT = ONNXRunTime
using Unicode: normalize
using WordTokenizers: TokenBuffer, isdone, flush!, character, spaces, atoms
using StringViews
import DoubleArrayTries
using DoubleArrayTries: DoubleArrayTrie
const DAT = DoubleArrayTries
using DataDeps

include("tokenizer.jl")
include("wordpiece.jl")
include("encoder.jl")
include("loader.jl")

export RankerModel, RankResult
## export rank # not exported to avoid name clash with PromptingTools
include("ranking.jl")

export EmbedderModel, EmbedResult
include("embedding.jl")

function __init__()
    ## Ranking models
    ## Acknowledgement: The weights come from Prithiviraj Damodaran's HF space: https://huggingface.co/prithivida/flashrank/tree/main
    register(DataDep("ms-marco-TinyBERT-L-2-v2",
        """
        TinyBERT-L-2-v2 cross-encoder trained on the ms-marco dataset.
        """,
        "https://huggingface.co/prithivida/flashrank/resolve/main/ms-marco-TinyBERT-L-2-v2.zip";
        post_fetch_method = unpack
    ))
    register(DataDep("ms-marco-MiniLM-L-12-v2",
        """
        MiniLM-L-12-v2 cross-encoder trained on the ms-marco dataset.
        """,
        "https://huggingface.co/prithivida/flashrank/resolve/main/ms-marco-MiniLM-L-12-v2.zip";
        post_fetch_method = unpack
    ))
    register(DataDep("ms-marco-MiniLM-L-4-v2",
        """
        MiniLM-L-4-v2 cross-encoder trained on the ms-marco dataset, FP32 precision.
        """,
        "https://huggingface.co/svilupp/onnx-cross-encoders/resolve/main/ms-marco-MiniLM-L-4-v2-onnx.zip";
        post_fetch_method = unpack
    ))
    register(DataDep("ms-marco-MiniLM-L-6-v2",
        """
        MiniLM-L-6-v2 cross-encoder trained on the ms-marco dataset, FP32 precision.
        """,
        "https://huggingface.co/svilupp/onnx-cross-encoders/resolve/main/ms-marco-MiniLM-L-6-v2-onnx.zip";
        post_fetch_method = unpack
    ))
    ## Embedding models - for acknowledgement, see the model cards in the archives and on HuggingFace
    register(DataDep("base-TinyBERT-L-4-v2",
        """
        TinyBERT-L-4-v2 model used for mean pooled embeddings.
        """,
        "https://huggingface.co/svilupp/onnx-embedders/resolve/main/TinyBERT_L-4_H-312_v2-onnx.zip",
        "04cc21a09f4675d07a1a12c02b9482f58d2087a8bda6825bf86289efced6582b";
        post_fetch_method = unpack
    ))
end

end
