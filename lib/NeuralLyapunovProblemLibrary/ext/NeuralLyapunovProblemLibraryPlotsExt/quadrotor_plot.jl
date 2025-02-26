function NeuralLyapunovProblemLibrary.plot_quadrotor_planar(x, y, θ, p, t; title="")
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
    function rotors(x, y, θ, r)
        pos = [x, y]
        r_vec = [r * cos(θ), r * sin(θ)] * 0.75
        r_perp = [-r_vec[2], r_vec[1]]
        top_arrow_begin = pos + r_vec
        top_arrow_end = pos + r_vec + r_perp
        bottom_arrow_begin = pos - r_vec
        bottom_arrow_end = pos - r_vec + r_perp
        top_arrow = [top_arrow_begin, top_arrow_end]
        bottom_arrow = [bottom_arrow_begin, bottom_arrow_end]
        return (
            (first.(top_arrow), last.(top_arrow)),
            (first.(bottom_arrow), last.(bottom_arrow))
        )
    end

    r = p[end]
    return @animate for i in eachindex(t)
        # Quadcopter body
        plot(quadrotor_body(x[i], y[i], θ[i], r), aspect_ratio=1, legend=false)

        # Rotors
        top_rotor, bottom_rotor = rotors(x[i], y[i], θ[i], r)
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
