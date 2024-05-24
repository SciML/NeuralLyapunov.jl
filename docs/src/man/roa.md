# Training for Region of Attraction Identification

Satisfying the [minimization](minimization.md) and [decrease](decrease.md) conditions within the training region (or any region around the fixed point, however small) is sufficient for proving local stability.
In many cases, however, we desire an estimate of the region of attraction, rather than simply a guarantee of local stability.

Any compact sublevel set wherein the minimization and decrease conditions are satisfied is an inner estimate of the region of attraction.
Therefore, we can restrict training for those conditions to only within a predetermined sublevel set ``\{ x : V(x) \le \rho \}``.
To do so, define a [`LyapunovDecreaseCondition`](@ref) as usual and then pass it through the [`make_RoA_aware`](@ref) function, which returns an analogous [`RoAAwareDecreaseCondition`](@ref).

```@docs
make_RoA_aware
RoAAwareDecreaseCondition
```
