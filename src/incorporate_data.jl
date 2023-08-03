"""
additional_loss_from_data(data, spec, network_func; fixed_point, jac, J_net, λ_minimization, λ_decrease)

Creates an additional loss function for use with NeuralPDE.PhysicsInformedNN
which uses data instead of an approximate dynamics model.

NeuralLyapunovPDESystem requires dynamics ẋ = f(x, p, t) and uses f when 
computing V̇. The loss function created by this function instead incorporates
data, which is a list of points (x, ẋ), calculating loss from the conditions
in spec at all points x and replacing the dynamics with ẋ at that point. The
average violation of the minimzation condition is multiplied by λ_minimization
and the average violation of the decrease condition is multiplied by λ_decrease
and the sum is returned by the loss function.
"""
function additional_loss_from_data(
    data, 
    spec::NeuralLyapunovSpecification, 
    network_func; 
    fixed_point,
    jac = ForwardDiff.jacobian,
    J_net = (_phi, _θ, x) -> jac((y) -> network_func(_phi, _θ, y), x),
    λ_minimization = 1.0,
    λ_decrease = 1.0,
    )
    # Unpack specifications and conditions
    structure = spec.structure
    minimzation_condition = spec.minimzation_condition
    decrease_condition = spec.decrease_condition

    _check_nonnegativity = check_nonnegativity(minimzation_condition)
    _check_decrease = check_decrease(decrease_condition)

    # Construct and return additional loss function for use with NeuralPDE
    if _check_nonnegativity && _check_decrease
        min_cond = get_minimization_condition(minimzation_condition)
        decrease_cond = get_decrease_condition(decrease_condition)

        return function(phi, θ, p)
            # Define network and Lyapunov function
            _net_func = (x) -> network_func(phi, θ, x)
            _J_net = (x) -> J_net(phi, θ, x)
            V = (x) -> structure.V(_net_func, x, fixed_point)

            return sum(
                function (data_point)
                    x, ẋ = data_point

                    # Conditions expect the dynamics to be a function of the 
                    # state, so we define a new V̇ for each x with dynamics 
                    # that always returns ẋ
                    V̇ = (_x) -> structure.V̇(
                        _net_func, 
                        _J_net, 
                        y -> ẋ, 
                        _x, 
                        fixed_point
                        )

                    min_loss = min_cond(V, x, fixed_point)^2
                    decrease_loss = decrease_cond(V, V̇, x, fixed_point)^2
                    
                    return λ_minimization * min_loss + λ_decrease * decrease_loss
                end,
                data
            ) / length(data)
        end
    elseif _check_nonnegativity
        min_cond = get_minimization_condition(minimzation_condition)

        return function(phi, θ, p)
            # Define network and Lyapunov function
            _net_func = (x) -> network_func(phi, θ, x)
            V = (x) -> structure.V(_net_func, x, fixed_point)

            return λ_minimization * sum(
                    function (data_point)
                        return min_cond(V, first(data_point), fixed_point)^2
                    end,
                    data
                ) / length(data)
        end
    elseif _check_decrease
        decrease_cond = get_decrease_condition(decrease_condition)

        return function(phi, θ, p)
            # Define network and Lyapunov function
            _net_func = (x) -> network_func(phi, θ, x)
            _J_net = (x) -> J_net(phi, θ, x)
            V = (x) -> structure.V(_net_func, x, fixed_point)
            
            return λ_decrease * sum(
                function (data_point)
                    x, ẋ = data_point

                    # Conditions expect the dynamics to be a function of the 
                    # state, so we define a new V̇ for each x with dynamics 
                    # that always returns ẋ
                    V̇ = (_x) -> structure.V̇(
                        _net_func, 
                        _J_net, 
                        y -> ẋ, 
                        _x, 
                        fixed_point
                        )

                    return decrease_cond(V, V̇, x, fixed_point)^2
                end,
                data
            ) / length(data)
        end
    else
        @warn "Additional loss function has no effect since neither the minimzation nor the decrease condition is checked."
        return (phi, θ, p) -> 0.0
    end
end