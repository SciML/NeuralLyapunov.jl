"""
    plot_quadrotor_planar(x, y, θ, [u1, u2,] p, t; title)
    plot_quadrotor_planar(sol, p; title, N, x_symbol, y_symbol, θ_symbol)

Plot the planar quadrotor's trajectory.

# Arguments
  - `x`: The x-coordinate of the quadrotor at each time step.
  - `y`: The y-coordinate of the quadrotor at each time step.
  - `θ`: The angle of the quadrotor at each time step.
  - `u1`: The thrust of the first rotor at each time step.
  - `u2`: The thrust of the second rotor at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the quadrotor.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `x`, `y`, `θ`, and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `x_symbol`: The symbol of the x-coordinate in `sol`; defaults to `:x`.
  - `y_symbol`: The symbol of the y-coordinate in `sol`; defaults to `:y`.
  - `θ_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
  - `u1_symbol`: The symbol of the thrust of the first rotor in `sol`; defaults to `:u1`.
  - `u2_symbol`: The symbol of the thrust of the second rotor in `sol`; defaults to `:u2`.
"""
function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(
    sol,
    p;
    title="",
    N = 500,
    x_symbol=:x,
    y_symbol=:y,
    θ_symbol=:θ,
    u1_symbol=:u1,
    u2_symbol=:u2
)
    t = LinRange(sol.t[1], sol.t[end], N)
    x = sol(t)[x_symbol]
    y = sol(t)[y_symbol]
    θ = sol(t)[θ_symbol]
    u1 = sol(t)[u1_symbol]
    u2 = sol(t)[u2_symbol]
    return plot_quadrotor_planar(x, y, θ, u1, u2, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(x, y, θ, p, t; title="")
    m, _, g, _ = p
    T = m * g / 2
    u = fill(T, length(t))
    return plot_quadrotor_planar(x, y, θ, u, u, p, t; title=title)
end

function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(x, y, θ, u1, u2, p, t; title="")
    function quadrotor_body(x, y, θ, r; aspect_ratio=10)
        pos = [x, y]
        r_vec = [r * cos(θ), r * sin(θ)]
        r_perp = [r_vec[2], -r_vec[1]] / aspect_ratio
        C1 = pos + r_vec + r_perp
        C2 = pos + r_vec - r_perp
        C3 = pos - r_vec - r_perp
        C4 = pos - r_vec + r_perp
        corners = [C1, C2, C3, C4]
        return Shape(first.(corners), last.(corners))
    end
    function rotors(x, y, θ, u1, u2, r)
        pos = [x, y]
        r_vec = [r * cos(θ), r * sin(θ)] * 0.75
        r_perp = [-r_vec[2], r_vec[1]]
        arrow1_begin = pos + r_vec
        arrow1_end = pos + r_vec + r_perp * u1
        arrow2_begin = pos - r_vec
        arrow2_end = pos - r_vec + r_perp * u2
        arrow1 = [arrow1_begin, arrow1_end]
        arrow2 = [arrow2_begin, arrow2_end]
        return (
            (first.(arrow1), last.(arrow1)),
            (first.(arrow2), last.(arrow2))
        )
    end

    m, _, g, r = p
    u1 = u1 ./ (m * g / 2)
    u2 = u2 ./ (m * g / 2)
    return @animate for i in eachindex(t)
        # Quadcopter body
        plot(quadrotor_body(x[i], y[i], θ[i], r), aspect_ratio=1, legend=false)

        # Rotors
        top_rotor, bottom_rotor = rotors(x[i], y[i], θ[i], u1[i], u2[i], r)
        plot!(top_rotor..., arrow=true, linewdith=2, color=:red)
        plot!(bottom_rotor..., arrow=true, linewdith=2, color=:red)

        # Trajectory so far
        traj_x = x[1:i]
        traj_y = y[1:i]
        plot!(traj_x, traj_y, color=:orange)
        scatter!(
            traj_x,
            traj_y,
            color = :orange,
            markersize = 2,
            markerstrokewidth = 0,
            markerstrokecolor = :orange,
        )

        # Plot settings and timestamp
        title!(title)
        #annotate!(-0.5L, 0.75L, "time= $(rpad(round(t[i]; digits=1),4,"0"))")
    end
end
