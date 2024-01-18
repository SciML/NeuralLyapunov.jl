using NeuralLyapunov
using Documenter

DocMeta.setdocmeta!(NeuralLyapunov, :DocTestSetup, :(using NeuralLyapunov); recursive=true)

makedocs(;
    modules=[NeuralLyapunov],
    authors="Nicholas Klugman <13633349+nicholaskl97@users.noreply.github.com> and contributors",
    sitename="NeuralLyapunov.jl",
    format=Documenter.HTML(;
        canonical="https://nicholaskl97.github.io/NeuralLyapunov.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nicholaskl97/NeuralLyapunov.jl",
    devbranch="main",
)
