"""
    AdditiveLyapunovNet(ϕ; ψ, m, r, dim_ϕ, dim_m, fixed_point)

Construct a Lyapunov-Net with the following structure:
```math
    V(x) = ψ(ϕ(x) - ϕ(x_0)) + r(m(x) - m(x_0)),
```
where ``x_0`` is `fixed_point` and the functions are defined as below.
If the functions meet the conditions listed below, the resulting model will be positive
definite (around `fixed_point`), as the ``r`` term will be positive definite and the ``ψ``
term will be positive semidefinite.

# Arguments
  - `ϕ`: The base neural network model; its output dimension should be `dim_ϕ`.
  - `ψ`: A Lux layer representing a positive semidefinite function that maps the output of
    `ϕ` to a scalar value; defaults to `SoSPooling()` (i.e., ``\\lVert ⋅ \\rVert^2``). Users
    may provide a function instead of a Lux layer, in which case it will be wrapped into a
    layer via `WrappedFunction`.
  - `m`: Optional pre-processing layer for use before `r`. This layer should output a vector
    of dimension `dim_m` and ``m(x) = m(x_0)`` should imply that ``x`` is an equilibrium to
    be analyzed by the Lyapunov function. Defaults to `NoOpLayer()`, which is typically the
    right choice when analyzing a single equilibrium point. Consider using a
    `Boltz.Layers.PeriodicEmbedding` if any of the state variables are periodic. Users may
    provide a function instead of a Lux layer, in which case it will be wrapped into a layer
    via `WrappedFunction`.
  - `r`: A Lux layer representing a positive definite function that maps the output of `m`
    to a scalar value; defaults to `SoSPooling()` (i.e., ``\\lVert ⋅ \\rVert^2``). Users may
    provide a function instead of a Lux layer, in which case it will be wrapped into a layer
    via `WrappedFunction`.
  - `dim_ϕ`: The dimension of the output of `ϕ`.
  - `dim_m`: The dimension of the output of `m`; defaults to `length(fixed_point)` when
    `fixed_point` is provided and `dim_m` isn't. Users must provide at least one of `dim_m`
    and `fixed_point`.
  - `fixed_point`: A vector in `ℝ^dim_m` representing the fixed point; defaults to
    `zeros(dim_m)` when `dim_m` is provided and `fixed_point` isn't. Users must provide at
    least one of `dim_m` and `fixed_point`.
"""
function AdditiveLyapunovNet(
    ϕ;
    ψ=SoSPooling(),
    m=NoOpLayer(),
    r=SoSPooling(),
    dim_ϕ,
    kwargs...
)
    if :dim_m in keys(kwargs)
        dim_m = kwargs[:dim_m]
        if :fixed_point in keys(kwargs)
            fixed_point = kwargs[:fixed_point]
        else
            fixed_point = zeros(dim_m)
        end
    elseif :fixed_point in keys(kwargs)
        fixed_point = kwargs[:fixed_point]
        dim_m = length(fixed_point)
    else
        throw(ArgumentError("Either `dim_m` or `fixed_point` must be provided."))
    end

    if ψ isa Function
        ψ = WrappedFunction(ψ)
    end
    if m isa Function
        m = WrappedFunction(m)
    end
    if r isa Function
        r = WrappedFunction(r)
    end

    return Parallel(
        +,
        Chain(
            ShiftTo(
                ϕ,
                fixed_point,
                zeros(eltype(fixed_point), dim_ϕ)
            ),
            ψ
        ),
        Chain(
            ShiftTo(
                m,
                fixed_point,
                zeros(eltype(fixed_point), dim_m)
            ),
            r
        )
    )
end

"""
    MultiplicativeLyapunovNet(ϕ; ζ, m, r, dim_m, fixed_point)
Construct a Lyapunov-Net with the following structure:
```math
    V(x) = ζ(ϕ(x)) (r(m(x) - m(x_0))),
```
where ``x_0`` is `fixed_point` and the functions are defined as below.
If the functions meet the conditions listed below, the resulting model will be positive
definite (around `fixed_point`), as the ``r`` term will be positive definite and the ``ζ``
term will be strictly positive.

# Arguments
  - `ϕ`: The base neural network model.
  - `ζ`: A Lux layer representing a strictly positive function that maps the output of
    `ϕ` to a scalar value; defaults to `StrictlyPositiveSoSPooling()` (i.e.,
    ``1 + \\lVert ⋅ \\rVert^2``). Users may provide a function instead of a Lux layer, in
    which case it will be wrapped into a layer via `WrappedFunction`.
  - `m`: Optional pre-processing layer for use before `r`. This layer should output a vector
    of dimension `dim_m` and ``m(x) = m(x_0)`` should imply that ``x`` is an equilibrium to
    be analyzed by the Lyapunov function. Defaults to `NoOpLayer()`, which is typically the
    right choice when analyzing a single equilibrium point. Consider using a
    `Boltz.Layers.PeriodicEmbedding` if any of the state variables are periodic. Users may
    provide a function instead of a Lux layer, in which case it will be wrapped into a layer
    via `WrappedFunction`.
  - `r`: A Lux layer representing a positive definite function that maps the output of `m`
    to a scalar value; defaults to `SoSPooling()` (i.e., ``\\lVert ⋅ \\rVert^2``). Users may
    provide a function instead of a Lux layer, in which case it will be wrapped into a layer
    via `WrappedFunction`.
  - `dim_m`: The dimension of the output of `m`; defaults to `length(fixed_point)` when
    `fixed_point` is provided and `dim_m` isn't.
  - `fixed_point`: A vector in `ℝ^dim_m` representing the fixed point; defaults to
    `zeros(dim_m)` when `dim_m` is provided and `fixed_point` isn't.
"""
function MultiplicativeLyapunovNet(
    ϕ;
    ζ=StrictlyPositiveSoSPooling(),
    m=NoOpLayer(),
    r=SoSPooling(),
    kwargs...
)
    if :dim_m in keys(kwargs)
        dim_m = kwargs[:dim_m]
        if :fixed_point in keys(kwargs)
            fixed_point = kwargs[:fixed_point]
        else
            fixed_point = zeros(dim_m)
        end
    elseif :fixed_point in keys(kwargs)
        fixed_point = kwargs[:fixed_point]
        dim_m = length(fixed_point)
    else
        throw(ArgumentError("Either `dim_m` or `fixed_point` must be provided."))
    end

    if ζ isa Function
        ζ = WrappedFunction(ζ)
    end
    if m isa Function
        m = WrappedFunction(m)
    end
    if r isa Function
        r = WrappedFunction(r)
    end

    return Parallel(
        .*,
        Chain(ϕ,ζ),
        Chain(
            ShiftTo(
                m,
                fixed_point,
                zeros(eltype(fixed_point), dim_m)
            ),
            r
        )
    )
end

"""
    SoSPooling(; dim = 1)

Construct a pooling function that computes the sum of squares along the dimension `dim`.
"""
SoSPooling(; dim = 1) = WrappedFunction(x -> sum(abs2, x, dims = dim))

"""
    StrictlyPositiveSoSPooling(; dim = 1)

Construct a pooling function that computes 1 + the sum of squares along the dimension `dim`.
"""
function StrictlyPositiveSoSPooling(; dim = 1)
    return WrappedFunction(x -> one(eltype(x)) .+ sum(abs2, x, dims = dim))
end
