```@meta
CurrentModule = NeuralLyapunov
```

# NeuralLyapunov.jl

[NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl) is a library for searching for neural Lyapunov functions in Julia.

This package provides an API for setting up the search for a neural Lyapunov function. Such a search can be formulated as a partial differential inequality, and this library generates a [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) PDESystem to be solved using [NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/). Since the Lyapunov conditions can be formulated in several different ways and a neural Lyapunov function can be set up in many different forms, this library presents an extensible interface for users to choose how they wish to set up the search, with useful pre-built options for common setups.

## Getting Started

If this is your first time using the library, start by familiarizing yourself with the [components of a neural Lyapunov problem](man.md) in NeuralLyapunov.jl.
Then, you can dive in with any of the following demonstrations (the [damped simple harmonic oscillator](demos/damped_SHO.md) is recommended to begin):

```@contents
Pages = Main.DEMONSTRATION_PAGES
Depth = 1
```

When you begin to write your own neural Lyapunov code, especially if you hope to define your own neural Lyapunov formulation, you may find any of the following manual pages useful:

```@contents
Pages = Main.MANUAL_PAGES
```

Finally, when you wish to test your neural Lyapunov code, you may wish to use one of the built-in example systems:

```@contents
Pages = Main.NEURALLYAPUNOVPROBLEMLIBRARY_PAGES
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:

      + The #diffeq-bridged and #sciml-bridged channels in the [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" *
                name *
                ".jl/tree/gh-pages/v" *
                version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" *
               name *
               ".jl/tree/gh-pages/v" *
               version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
