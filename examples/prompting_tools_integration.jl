## Example of how to integrate FlashRank.jl into your [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) RAG pipeline:

using LinearAlgebra, SparseArrays, Unicode # imports required for full PT functionality
using PromptingTools
using PromptingTools.Experimental.RAGTools
const RT = PromptingTools.Experimental.RAGTools
using FlashRank

# Define a type of AbstractReranker
struct FlashRanker <: RT.AbstractReranker
    model::RankerModel
end

# Define the method for reranking with it
function RT.rerank(
        reranker::FlashRanker, index::RT.AbstractDocumentIndex, question::AbstractString,
        candidates::RT.AbstractCandidateChunks;
        verbose::Bool = false,
        top_n::Integer = length(candidates.scores),
        kwargs...)
    @assert top_n>0 "top_n must be a positive integer."
    documents = index[candidates, :chunks]
    @assert !(isempty(documents)) "The candidate chunks must not be empty for Cohere Reranker! Check the index IDs."

    ## Run re-ranker
    ranker = reranker.model
    result = ranker(question, documents; top_n)

    ## Unwrap re-ranked positions
    scores = result.scores
    positions = candidates.positions[result.positions]
    index_ids = if candidates isa MultiCandidateChunks
        candidates.index_ids[result.positions]
    else
        candidates.index_id
    end

    verbose && @info "Reranking done in $(round(res.elapsed; digits=1)) seconds."

    return candidates isa RT.MultiCandidateChunks ?
           RT.MultiCandidateChunks(index_ids, positions, scores) :
           RT.CandidateChunks(index_ids, positions, scores)
end

## Sample data
sentences = [
    "Search for the latest advancements in quantum computing using Julia language.",
    "How to implement machine learning algorithms in Julia with examples.",
    "Looking for performance comparison between Julia, Python, and R for data analysis.",
    "Find Julia language tutorials focusing on high-performance scientific computing.",
    "Search for the top Julia language packages for data visualization and their documentation.",
    "How to set up a Julia development environment on Windows 10.",
    "Discover the best practices for parallel computing in Julia.",
    "Search for case studies of large-scale data processing using Julia.",
    "Find comprehensive resources for mastering metaprogramming in Julia.",
    "Looking for articles on the advantages of using Julia for statistical modeling.",
    "How to contribute to the Julia open-source community: A step-by-step guide.",
    "Find the comparison of numerical accuracy between Julia and MATLAB.",
    "Looking for the latest Julia language updates and their impact on AI research.",
    "How to efficiently handle big data with Julia: Techniques and libraries.",
    "Discover how Julia integrates with other programming languages and tools.",
    "Search for Julia-based frameworks for developing web applications.",
    "Find tutorials on creating interactive dashboards with Julia.",
    "How to use Julia for natural language processing and text analysis.",
    "Discover the role of Julia in the future of computational finance and econometrics."
]
## Build the index
index = build_index(
    sentences; chunker_kwargs = (; sources = map(i -> "Doc$i", 1:length(sentences))))

# Wrap the model to be a valid Ranker recognized by RAGTools
# It will be provided to the airag/rerank function to avoid instantiating it on every call
reranker = RankerModel(:tiny) |> FlashRanker

## Apply to the pipeline configuration, eg, 
cfg = RAGConfig(; retriever = AdvancedRetriever(; reranker))

# Ask a question
question = "What are the best practices for parallel computing in Julia?"
result = airag(cfg, index; question, return_all = true)
