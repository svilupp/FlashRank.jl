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

function __init__()
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
end

end
