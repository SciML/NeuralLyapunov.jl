var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralLyapunov","category":"page"},{"location":"#NeuralLyapunov","page":"Home","title":"NeuralLyapunov","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NeuralLyapunov.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [NeuralLyapunov]","category":"page"},{"location":"#NeuralLyapunov.AbstractLyapunovDecreaseCondition","page":"Home","title":"NeuralLyapunov.AbstractLyapunovDecreaseCondition","text":"AbstractLyapunovDecreaseCondition\n\nRepresents the decrease condition in a neural Lyapunov problem\n\nAll concrete AbstractLyapunovDecreaseCondition subtypes should define the check_decrease and get_decrease_condition functions.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.AbstractLyapunovMinimizationCondition","page":"Home","title":"NeuralLyapunov.AbstractLyapunovMinimizationCondition","text":"AbstractLyapunovMinimizationCondition\n\nRepresents the minimization condition in a neural Lyapunov problem\n\nAll concrete AbstractLyapunovMinimizationCondition subtypes should define the check_nonnegativity, check_fixed_point, and get_minimization_condition functions.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.LyapunovDecreaseCondition","page":"Home","title":"NeuralLyapunov.LyapunovDecreaseCondition","text":"LyapunovDecreaseCondition(check_decrease, decrease, strength, rectifier)\n\nSpecifies the form of the Lyapunov conditions to be used; if check_decrease, training will enforce decrease(V, dVdt) ≤ strength(state, fixed_point).\n\nThe inequality will be approximated by the equation     rectifier(decrease(V, dVdt) - strength(state, fixed_point)) = 0.0.\n\nIf the dynamics truly have a fixed point at fixed_point and dVdt has been defined properly in terms of the dynamics, then dVdt(fixed_point) will be 0 and there is no need to enforce dVdt(fixed_point) = 0, so check_fixed_point defaults to false.\n\nExamples:\n\nAsymptotic decrease can be enforced by requiring     dVdt ≤ -C |state - fixed_point|^2, which corresponds to     decrease = (V, dVdt) -> dVdt     strength = (x, x0) -> -C * (x - x0) ⋅ (x - x0)\n\nExponential decrease of rate k is proven by dVdt ≤ - k * V, so corresponds to     decrease = (V, dVdt) -> dVdt + k * V     strength = (x, x0) -> 0.0\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.LyapunovMinimizationCondition","page":"Home","title":"NeuralLyapunov.LyapunovMinimizationCondition","text":"LyapunovMinimizationCondition\n\nSpecifies the form of the Lyapunov conditions to be used.\n\nIf check_nonnegativity is true, training will attempt to enforce     V(state) ≥ strength(state, fixed_point). The inequality will be approximated by the equation     rectifier(strength(state, fixed_point) - V(state)) = 0.0. If check_fixed_point is true, then training will also attempt to enforce     V(fixed_point) = 0.\n\nExamples\n\nThe condition that the Lyapunov function must be minimized uniquely at the fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when state ≠ fixed_point. This could be enfored by V(fixed_point) ≥ ||state - fixed_point||^2, which would be represented, with check_nonnegativity = true, by     strength(state, fixedpoint) = ||state - fixedpoint||^2, paired with V(fixed_point) = 0, which can be enforced with check_fixed_point = true.\n\nIf V were structured such that it is always nonnegative, then V(fixed_point) = 0 is all that must be enforced in training for the Lyapunov function to be uniquely minimized at fixed_point. So, in that case, we would use     check_nonnegativity = false;  check_fixed_point = true.\n\nIn either case, rectifier = (t) -> max(0.0, t) exactly represents the inequality, but differentiable approximations of this function may be employed.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.NeuralLyapunovSpecification","page":"Home","title":"NeuralLyapunov.NeuralLyapunovSpecification","text":"NeuralLyapunovSpecification\n\nSpecifies a neural Lyapunov problem\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.NeuralLyapunovStructure","page":"Home","title":"NeuralLyapunov.NeuralLyapunovStructure","text":"NeuralLyapunovStructure\n\nSpecifies the structure of the neural Lyapunov function and its derivative.\n\nAllows the user to define the Lyapunov in terms of the neural network to structurally enforce Lyapunov conditions. V(phi::Function, state, fixed_point) takes in the neural network, the state, and the fixed point, and outputs the value of the Lyapunov function at state. V̇(phi::Function, J_phi::Function, f::Function, state, params, t, fixed_point) takes in the neural network, jacobian of the neural network, dynamics, state, parameters and time (for calling the dynamics, when relevant), and fixed point, and outputs the time derivative of the Lyapunov function at state. f_call(dynamics::Function, phi::Function, state, p, t) takes in the dynamics, the neural network, the state, the parameters of the dynamics, and time, and outputs the derivative of the state; this is useful for making closed-loop dynamics which depend on the neural network, such as in the policy search case network_dim is the dimension of the output of the neural network.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.RoAAwareDecreaseCondition","page":"Home","title":"NeuralLyapunov.RoAAwareDecreaseCondition","text":"RoAAwareDecreaseCondition(check_decrease, decrease, strength, rectifier, ρ,\n                          out_of_RoA_penalty)\n\nSpecifies the form of the Lyapunov conditions to be used, training for a region of attraction estimate of { x : V(x) ≤ ρ }\n\nIf check_decrease, training will enforce decrease(V(state), dVdt(state)) ≤ strength(state, fixed_point) whenever V(state) ≤ ρ, and will instead apply |out_of_RoA_penalty(V(state), dVdt(state), state, fixed_point, ρ)|^2 when V(state) > ρ.\n\nThe inequality will be approximated by the equation     rectifier(decrease(V(state), dVdt(state)) - strength(state, fixed_point)) = 0.0.\n\nIf the dynamics truly have a fixed point at fixed_point and dVdt has been defined properly in terms of the dynamics, then dVdt(fixed_point) will be 0 and there is no need to enforce dVdt(fixed_point) = 0, so check_fixed_point defaults to false.\n\nExamples:\n\nAsymptotic decrease can be enforced by requiring     dVdt ≤ -C |state - fixed_point|^2, which corresponds to     decrease = (V, dVdt) -> dVdt and     strength = (x, x0) -> -C * (x - x0) ⋅ (x - x0).\n\nExponential decrease of rate k is proven by dVdt ≤ - k * V, so corresponds to     decrease = (V, dVdt) -> dVdt + k * V and     strength = (x, x0) -> 0.0.\n\nEnforcing either condition only in the region of attraction and not penalizing any points outside that region would correspond to     out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 0.0, whereas an example of a penalty that decays farther in state space from the fixed point is     out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 1.0 / ((x - x0) ⋅ (x - x0)). Note that this penalty could also depend on values of V, dVdt, and ρ.\n\n\n\n\n\n","category":"type"},{"location":"#NeuralLyapunov.AsymptoticDecrease-Tuple{}","page":"Home","title":"NeuralLyapunov.AsymptoticDecrease","text":"AsymptoticDecrease(; strict, C, rectifier)\n\nConstruct a LyapunovDecreaseCondition corresponding to asymptotic decrease.\n\nIf strict is false, the condition is dV/dt ≤ 0, and if strict is true, the condition is dV/dt ≤ - C | state - fixed_point |^2.\n\nThe inequality is represented by a ≥ b <==> rectifier(b-a) = 0.0.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.DontCheckDecrease-Tuple{}","page":"Home","title":"NeuralLyapunov.DontCheckDecrease","text":"DontCheckDecrease()\n\nConstruct a LyapunovDecreaseCondition which represents not checking for decrease of the Lyapunov function along system trajectories. This is appropriate in cases when the Lyapunov decrease condition has been structurally enforced.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.DontCheckNonnegativity-Tuple{}","page":"Home","title":"NeuralLyapunov.DontCheckNonnegativity","text":"DontCheckNonnegativity(check_fixed_point)\n\nConstruct a LyapunovMinimizationCondition which represents not checking for nonnegativity of the Lyapunov function. This is appropriate in cases where this condition has been structurally enforced.\n\nIt is still possible to check for V(fixed_point) = 0, even in this case, for example if V is structured to be positive for state ≠ fixed_point, but it is not guaranteed structurally that V(fixed_point) = 0.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.ExponentialDecrease-Tuple{Real}","page":"Home","title":"NeuralLyapunov.ExponentialDecrease","text":"ExponentialDecrease(k; strict, C, rectifier)\n\nConstruct a LyapunovDecreaseCondition corresponding to exponential decrease of rate k.\n\nIf strict is false, the condition is dV/dt ≤ -k * V, and if strict is true, the condition is dV/dt ≤ -k * V - C * ||state - fixed_point||^2.\n\nThe inequality is represented by a ≥ b <==> rectifier(b-a) = 0.0.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.NeuralLyapunovPDESystem-Tuple{Function, Any, Any, NeuralLyapunovSpecification}","page":"Home","title":"NeuralLyapunov.NeuralLyapunovPDESystem","text":"NeuralLyapunovPDESystem(dynamics::ODESystem, bounds, spec; <keyword_arguments>)\nNeuralLyapunovPDESystem(dynamics::Function, lb, ub, spec; <keyword_arguments>)\n\nConstruct a ModelingToolkit.PDESystem representing the specified neural Lyapunov problem.\n\nArguments\n\ndynamics: the dynamical system being analyzed, represented as an ODESystem or the       function f such that ẋ = f(x[, u], p, t); either way, the ODE should not depend       on time and only t = 0.0 will be used\nbounds: an array of domains, defining the training domain by bounding the states (and       derivatives, when applicable) of dynamics; only used when dynamics isa       ODESystem, otherwise use lb and ub.\nlb and ub: the training domain will be lb_1 ub_1lb_2 ub_2; not used       when dynamics isa ODESystem, then use bounds.\nspec::NeuralLyapunovSpecification: defines the Lyapunov function structure, as well as       the minimization and decrease conditions.\nfixed_point: the equilibrium being analyzed; defaults to the origin.\np: the values of the parameters of the dynamical system being analyzed; defaults to       SciMLBase.NullParameters(); not used when dynamics isa ODESystem, then use the       default parameter values of dynamics.\nstate_syms: an array of the Symbol representing each state; not used when dynamics       isa ODESystem, then the symbols from dynamics are used; if dynamics isa       ODEFunction, symbols stored there are used, unless overridden here; if not provided       here and cannot be inferred, [:state1, :state2, ...] will be used.\nparameter_syms: an array of the Symbol representing each parameter; not used when       dynamics isa ODESystem, then the symbols from dynamics are used; if dynamics       isa ODEFunction, symbols stored there are used, unless overridden here; if not       provided here and cannot be inferred, [:param1, :param2, ...] will be used.\npolicy_search::Bool: whether or not to include a loss term enforcing fixed_point to       actually be a fixed point; defaults to false; only used when dynamics isa       Function && !(dynamics isa ODEFunction); when dynamics isa ODEFunction,       policy_search must be false, so should not be supplied; when dynamics isa       ODESystem, value inferred by the presence of unbound inputs.\nname: the name of the constructed PDESystem\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.NonnegativeNeuralLyapunov-Tuple{Integer}","page":"Home","title":"NeuralLyapunov.NonnegativeNeuralLyapunov","text":"NonnegativeNeuralLyapunov(network_dim, δ, pos_def; grad_pos_def, grad)\n\nCreate a NeuralLyapunovStructure where the Lyapunov function is the L2 norm of the neural network output plus a constant δ times a function pos_def.\n\nThe condition that the Lyapunov function must be minimized uniquely at the fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when state ≠ fixed_point. This structure ensures V(state) ≥ 0. Further, if δ > 0 and pos_def(fixed_point, fixed_point) = 0, but pos_def(state, fixed_point) > 0 when state ≠ fixed_point, this ensures that V(state) > 0 when state != fixed_point. This does not enforce V(fixed_point) = 0, so that condition must included in the neural Lyapunov loss function.\n\ngrad_pos_def(state, fixed_point) should be the gradient of pos_def with respect to state at state. If grad_pos_def is not defined, it is evaluated using grad, which defaults to ForwardDiff.gradient.\n\nThe neural network output has dimension network_dim.\n\nDynamics are assumed to be in f(state, p, t) form, as in an ODEFunction. For f(state, input, p, t), consider using add_policy_search.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.PositiveSemiDefinite-Tuple{}","page":"Home","title":"NeuralLyapunov.PositiveSemiDefinite","text":"PositiveSemiDefinite(check_fixed_point)\n\nConstruct a LyapunovMinimizationCondition representing     V(state) ≥ 0. If check_fixed_point is true, then training will also attempt to enforce     V(fixed_point) = 0.\n\nThe inequality is represented by a ≥ b <==> rectifier(b-a) = 0.0.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.PositiveSemiDefiniteStructure-Tuple{Integer}","page":"Home","title":"NeuralLyapunov.PositiveSemiDefiniteStructure","text":"PositiveSemiDefiniteStructure(network_dim; pos_def, non_neg, grad_pos_def, grad_non_neg, grad)\n\nCreate a NeuralLyapunovStructure where the Lyapunov function is the product of a positive (semi-)definite function pos_def which does not depend on the network and a nonnegative function non_neg which does depend the network.\n\nThe condition that the Lyapunov function must be minimized uniquely at the fixed point can be represented as V(fixed_point) = 0, V(state) > 0 when state ≠ fixed_point. This structure ensures V(state) ≥ 0. Further, if pos_def is 0 only at fixed_point (and positive elsewhere) and if non_neg is strictly positive away from fixed_point (as is the case for the default values of pos_def and non_neg), then this structure ensures V(fixed_point) = 0 and V(state) > 0 when state ≠ fixed_point.\n\ngrad_pos_def(state, fixed_point) should be the gradient of pos_def with respect to state at state. Similarly, grad_non_neg(net, J_net, state, fixed_point) should be the gradient of non_neg(net, state, fixed_point) with respect to state at state. If grad_pos_def or grad_non_neg is not defined, it is evaluated using grad, which defaults to ForwardDiff.gradient.\n\nThe neural network output has dimension network_dim.\n\nDynamics are assumed to be in f(state, p, t) form, as in an ODEFunction. For f(state, input, p, t), consider using add_policy_search.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.StrictlyPositiveDefinite-Tuple{}","page":"Home","title":"NeuralLyapunov.StrictlyPositiveDefinite","text":"StrictlyPositiveDefinite(C; check_fixed_point, rectifier)\n\nConstruct a LyapunovMinimizationCondition representing     V(state) ≥ C * ||state - fixed_point||^2. If check_fixed_point is true, then training will also attempt to enforce     V(fixed_point) = 0.\n\nThe inequality is represented by a ≥ b <==> rectifier(b-a) = 0.0.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.UnstructuredNeuralLyapunov-Tuple{}","page":"Home","title":"NeuralLyapunov.UnstructuredNeuralLyapunov","text":"UnstructuredNeuralLyapunov()\n\nCreate a NeuralLyapunovStructure where the Lyapunov function is the neural network evaluated at the state. This does not structurally enforce any Lyapunov conditions.\n\nDynamics are assumed to be in f(state, p, t) form, as in an ODEFunction. For f(state, input, p, t), consider using add_policy_search.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.add_policy_search-Tuple{NeuralLyapunovStructure, Integer}","page":"Home","title":"NeuralLyapunov.add_policy_search","text":"add_policy_search(lyapunov_structure, new_dims, control_structure)\n\nAdd dependence on the neural network to the dynamics in a NeuralLyapunovStructure.\n\nAdd new_dims outputs to the neural network and feeds them through control_structure to calculate the contribution of the neural network to the dynamics. Use the existing lyapunov_structure.network_dim dimensions as in lyapunov_structure to calculate the Lyapunov function.\n\nlyapunov_structure should assume in its V̇ that the dynamics take a form f(x, p, t). The returned NeuralLyapunovStructure will assume instead f(x, u, p, t), where u is the contribution from the neural network. Therefore, this structure cannot be used with a NeuralLyapunovPDESystem method that requires an ODEFunction, ODESystem, or ODEProblem.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.check_decrease-Tuple{NeuralLyapunov.AbstractLyapunovDecreaseCondition}","page":"Home","title":"NeuralLyapunov.check_decrease","text":"check_decrease(cond::AbstractLyapunovDecreaseCondition)\n\nReturn true if cond specifies training to meet the Lyapunov decrease condition, and false if cond specifies no training to meet this condition.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.check_minimal_fixed_point-Tuple{NeuralLyapunov.AbstractLyapunovMinimizationCondition}","page":"Home","title":"NeuralLyapunov.check_minimal_fixed_point","text":"check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)\n\nReturn true if cond specifies training for the Lyapunov function to equal zero at the fixed point, and false if cond specifies no training to meet this condition.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.check_nonnegativity-Tuple{NeuralLyapunov.AbstractLyapunovMinimizationCondition}","page":"Home","title":"NeuralLyapunov.check_nonnegativity","text":"check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)\n\nReturn true if cond specifies training to meet the Lyapunov minimization condition, and false if cond specifies no training to meet this condition.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.get_decrease_condition-Tuple{NeuralLyapunov.AbstractLyapunovDecreaseCondition}","page":"Home","title":"NeuralLyapunov.get_decrease_condition","text":"get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)\n\nReturn a function of V, dVdt, state, and fixed_point that is equal to zero when the Lyapunov decrease condition is met and is greater than zero when it is violated.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.get_minimization_condition-Tuple{NeuralLyapunov.AbstractLyapunovMinimizationCondition}","page":"Home","title":"NeuralLyapunov.get_minimization_condition","text":"get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)\n\nReturn a function of V, state, and fixed_point that equals zero when the Lyapunov minimization condition is met and is greater than zero when it's violated.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.get_numerical_lyapunov_function-Tuple{Any, Any, NeuralLyapunovStructure, Function, Any}","page":"Home","title":"NeuralLyapunov.get_numerical_lyapunov_function","text":"get_numerical_lyapunov_function(phi, θ, structure, dynamics, fixed_point;\n                                <keyword_arguments>)\n\nCombine Lyapunov function structure, dynamics, and neural network weights to generate Julia functions representing the Lyapunov function and its time derivative: V(x) V(x).\n\nThese functions can operate on a state vector or columnwise on a matrix of state vectors.\n\nArguments\n\nphi: the neural network, represented as phi(x, θ) if the neural network has a single       output, or a Vector of the same with one entry per neural network output.\nθ: the parameters of the neural network; θ[:φ1] should be the parameters of the first       neural network output (even if there is only one), θ[:φ2] the parameters of the       second (if there are multiple), and so on.\nstructure: a NeuralLyapunovStructure representing the structure of the neural       Lyapunov function.\ndynamics: the system dynamics, as a function to be used in conjunction with       structure.f_call.\nfixed_point: the equilibrium point being analyzed by the Lyapunov function.\np: parameters to be passed into dynamics; defaults to SciMLBase.NullParameters().\nuse_V̇_structure: when true, V(x) is calculated using structure.V̇; when false,       V(x) is calculated using deriv as fracddt V(x + t f(x)) at       t = 0; defaults to false, as it is more efficient in many cases.\nderiv: a function for calculating derivatives; defaults to (and expects same arguments       as) ForwardDiff.derivative; only used when use_V̇_structure is false.\njac: a function for calculating Jacobians; defaults to (and expects same arguments as)       ForwardDiff.jacobian; only used when use_V̇_structure is true.\nJ_net: the Jacobian of the neural network, specified as a function       J_net(phi, θ, state); if isnothing(J_net) (as is the default), J_net will be       calculated using jac; only used when use_V̇_structure is true.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.get_policy-Tuple{Any, Any, Integer, Integer}","page":"Home","title":"NeuralLyapunov.get_policy","text":"get_policy(phi, θ, network_func, dim; control_structure)\n\nGenerate a Julia function representing the control policy as a function of the state\n\nThe returned function can operate on a state vector or columnwise on a matrix of state vectors.\n\nphi is the neural network with parameters θ. The network should have network_dim outputs, the last control_dim of which will be passed into control_structure to create the policy output.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.local_lyapunov-Union{Tuple{T}, Tuple{Function, Any, Any, AbstractMatrix{T}}} where T<:Number","page":"Home","title":"NeuralLyapunov.local_lyapunov","text":"get_local_lyapunov(dynamics, state_dim, optimizer_factory[, jac]; fixed_point, p)\n\nUse semidefinite programming to derive a quadratic Lyapunov function for the linearization of dynamics around fixed_point. Return (V, dV/dt).\n\nTo solve the semidefinite program, JuMP.Model requires an optimizer_factory capable of semidefinite programming (SDP). See the JuMP documentation for examples.\n\nIf jac is not supplied, the Jacobian of the dynamics(x, p, t) with respect to x is calculated using ForwardDiff. Otherwise, jac is expected to be either a function or an AbstractMatrix. If jac isa Function, it should take in the state and parameters and output the Jacobian of dynamics with respect to the state x. If jac isa AbstractMatrix, it should be the value of the Jacobian at fixed_point.\n\nIf fixed_point is not specified, it defaults to the origin, i.e., zeros(state_dim). Parameters p for the dynamics should be supplied when the dynamics depend on them.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.make_RoA_aware-Tuple{LyapunovDecreaseCondition}","page":"Home","title":"NeuralLyapunov.make_RoA_aware","text":"make_RoA_aware(cond; ρ, out_of_RoA_penalty)\n\nAdd awareness of the region of attraction (RoA) estimation task to the supplied LyapunovDecreaseCondition\n\nρ is the target level such that the RoA will be { x : V(x) ≤ ρ }. cond specifies the loss applied when V(state) ≤ ρ, and |out_of_RoA_penalty(V(state), dVdt(state), state, fixed_point, ρ)|^2 is the loss from state values such that V(state) > ρ.\n\n\n\n\n\n","category":"method"},{"location":"#NeuralLyapunov.phi_to_net-Tuple{Any, Any}","page":"Home","title":"NeuralLyapunov.phi_to_net","text":"phi_to_net(phi, θ[; idx])\n\nReturn the network as a function of state alone.\n\nArguments\n\nphi: the neural network, represented as phi(x, θ) if the neural network has a single       output, or a Vector of the same with one entry per neural network output.\nθ: the parameters of the neural network; θ[:φ1] should be the parameters of the first       neural network output (even if there is only one), θ[:φ2] the parameters of the       second (if there are multiple), and so on.\nidx: the neural network outputs to include in the returned function; defaults to all and       only applicable when phi isa Vector.\n\n\n\n\n\n","category":"method"}]
}
