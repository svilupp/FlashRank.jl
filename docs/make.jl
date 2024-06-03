using FlashRank
using Documenter

DocMeta.setdocmeta!(FlashRank, :DocTestSetup, :(using FlashRank); recursive = true)

makedocs(;
    modules = [FlashRank],
    authors = "J S <49557684+svilupp@users.noreply.github.com> and contributors",
    sitename = "FlashRank.jl",
    format = Documenter.HTML(;
        canonical = "https://svilupp.github.io/FlashRank.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api_reference.md"
    ]
)

deploydocs(;
    repo = "github.com/svilupp/FlashRank.jl",
    devbranch = "main"
)
