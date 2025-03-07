using NeuralLyapunov
using Documenter

DocMeta.setdocmeta!(
    NeuralLyapunov, :DocTestSetup, :(using NeuralLyapunov); recursive = true)

MANUAL_PAGES = [
    "man.md",
    "man/pdesystem.md",
    "man/minimization.md",
    "man/decrease.md",
    "man/structure.md",
    "man/roa.md",
    "man/policy_search.md",
    "man/local_lyapunov.md"
]
DEMONSTRATION_PAGES = [
    "demos/damped_SHO.md",
    "demos/roa_estimation.md",
    "demos/policy_search.md",
    "demos/benchmarking.md"
]

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
        "Home" => "index.md",
        "Manual" => vcat(MANUAL_PAGES, hide("man/internals.md")),
        "Demonstrations" => DEMONSTRATION_PAGES
    ]
)

deploydocs(;
    repo = "github.com/SciML/NeuralLyapunov.jl",
    devbranch = "master"
)
