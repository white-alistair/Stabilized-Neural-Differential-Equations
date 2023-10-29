begin
    using StabilizedNDEs: StabilizedNDE
    using OrdinaryDiffEq
    using CairoMakie

    function f(u, p, t)
        x, y = u
        return [y, -x]
    end

    function g(u, t)
        x, y = u
        return [x^2 + y^2 - 1]
    end

    function jacobian(u, t)
        x, y = u
        return [2x 2y]
    end

    function pseudoinverse(u, t)
        if u[1] == u[2] == 0
            return [0; 0;;]
        end
        J = jacobian(u, t)
        return J' * inv(J * J')
    end

    function stabilization_term(u, t)
        return -pseudoinverse(u, t) * g(u, t)
    end

    # Set up the figure
    size_inches = 1.5 .* (5, 2.1)
    size_pt = 72 .* size_inches
    ticksize = 4
    fig = Figure(resolution = size_pt, fontsize = 12, figure_padding = (8, 8, 0, 0))
    ax1 = Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect(),
        xticksize = ticksize,
        yticksize = ticksize,
    )
    hidedecorations!(ax1, grid = false)
    ax2 = Axis(
        fig[1, 2],
        xlabel = "x",
        aspect = DataAspect(),
        xticksize = ticksize,
        yticksize = ticksize,
    )
    hidedecorations!(ax2, grid = false)

    constraint_colour = :black
    nde_colour = :dodgerblue
    snde_colour = :orangered
    nde_traj_col = :purple
    snde_traj_colour = :seagreen

    linewidth = 2.5

    # Grid
    xs = -3:0.4:3
    ys = -3:0.4:3

    # Unstabilized vector field (first term in SNDE)
    u_f = [y for x in xs, y in ys]
    v_f = [-x for x in xs, y in ys]

    # Stabilization term (second term in SNDE)
    u_stab = [stabilization_term([x, y], nothing)[1] for x in xs, y in ys]
    v_stab = [stabilization_term([x, y], nothing)[2] for x in xs, y in ys]

    # Stabilized vector field (SNDE)
    γ = 0.66
    u_total = u_f .+ γ * u_stab
    v_total = v_f .+ γ * v_stab

    arc!(ax1, Point2f(0), 1, -π, π, linewidth = linewidth, color = constraint_colour)
    arrows!(
        ax1,
        xs,
        ys,
        u_f,
        v_f,
        linewidth = 1,
        arrowsize = 5,
        lengthscale = 0.13,
        color = nde_colour,
    )
    xlims!(ax1, -2, 2)
    ylims!(ax1, -2, 2)

    arc!(ax2, Point2f(0), 1, -π, π, linewidth = linewidth, color = constraint_colour)
    arrows!(
        ax2,
        xs,
        ys,
        u_total,
        v_total,
        linewidth = 1,
        arrowsize = 5,
        lengthscale = 0.13,
        color = snde_colour,
    )
    xlims!(ax2, -2, 2)
    ylims!(ax2, -2, 2)

    # Plot an unstabilized trajectory
    u0 = [1.2, 1.2]
    prob = ODEProblem{false}(f, u0, (0.0, 5.0))
    sol = solve(prob, Tsit5(), saveat = 0.01)
    traj = Array(sol)
    lines!(ax1, traj[1, :], traj[2, :], color = nde_traj_col, linewidth = linewidth)

    # Plot arrows along the trajectory
    points = traj[:, end:-90:50]
    directions = mapslices(u -> f(u, nothing, nothing), points, dims = 1)
    arrows!(
        ax1,
        points[1, :],
        points[2, :],
        directions[1, :],
        directions[2, :],
        lengthscale = 0.001,
        color = nde_traj_col,
        arrowsize = 14,
    )

    # Plot a stabilized trajectory
    snde = StabilizedNDE(f, γ, pseudoinverse, g)
    prob = ODEProblem{false}(snde, u0, (0.0, 7.0))
    sol = solve(prob, Tsit5(), saveat = 0.01)
    traj = Array(sol)
    lines!(ax2, traj[1, :], traj[2, :], color = snde_traj_colour, linewidth = linewidth)

    # Plot arrows along the trajectory
    points = traj[:, end:-90:50]
    directions = mapslices(u -> snde(u, nothing, nothing), points, dims = 1)
    arrows!(
        ax2,
        points[1, :],
        points[2, :],
        directions[1, :],
        directions[2, :],
        lengthscale = 0.001,
        color = snde_traj_colour,
        arrowsize = 14,
    )

    # Legend
    elem_constraint = [LineElement(; color = constraint_colour, linewidth)]
    elem_snde = [LineElement(; color = snde_colour, linewidth)]
    elem_node = [LineElement(; color = nde_colour, linewidth)]
    elem_nde_traj = [LineElement(; color = nde_traj_col, linewidth)]
    elem_snde_traj = [LineElement(; color = snde_traj_colour, linewidth)]
    Legend(
        fig[1, 3],
        [elem_constraint, elem_node, elem_nde_traj, elem_snde, elem_snde_traj],
        ["Constraint Manifold", "NDE", "NDE Trajectory", "SNDE", "SNDE Trajectory"],
        orientation = :vertical,
        tellwidth = true,
        tellheight = false,
        padding = 3,
    )
    Label(fig[1, 1, Top()], "(a)", font = :bold, padding = (0, 3, -20, 0))
    Label(fig[1, 2, Top()], "(b)", font = :bold, padding = (0, 3, -20, 0))
    fig
end

CairoMakie.save("figures/figure_1.pdf", fig)
