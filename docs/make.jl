using NeuralLyapunov
using Documenter

DocMeta.setdocmeta!(
    NeuralLyapunov, :DocTestSetup, :(using NeuralLyapunov); recursive = true)

makedocs(;
    modules = [NeuralLyapunov],
    authors = "Nicholas Klugman <13633349+nicholaskl97@users.noreply.github.com> and contributors",
    sitename = "NeuralLyapunov.jl",
    format = Documenter.HTML(;
        canonical = "https://SciML.github.io/NeuralLyapunov.jl",
        edit_link = "master",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/SciML/NeuralLyapunov.jl",
    devbranch = "master"
)
