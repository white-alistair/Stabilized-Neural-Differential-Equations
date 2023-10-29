begin
    using JLD2
    using CairoMakie
    using Statistics

    # 1. Get the data
    two_body_data = JLD2.load_object("figures/data/two_body_problem.jld2")
    hnn_data = JLD2.load_object("figures/data/two_body_problem_hnn.jld2")

    rigid_body_data = JLD2.load_object("figures/data/rigid_body.jld2")
    tpo_data = JLD2.load_object("figures/data/rigid_body_tpo.jld2")

    # 2. Set up the figure
    size_inches = 1.6 .* (5, 5)
    size_pt = 72 .* size_inches
    linewidth = 1
    fig = Figure(resolution = size_pt, fontsize = 12, figure_padding = 5)

    node_colour = :dodgerblue
    anode_colour = :darkgreen
    snde_colour = :orangered
    sanode_colour = :purple
    hnn_colour = :magenta
    tpo_colour = :navy

    # 2.1. First row: two body problem
    ax1 = Axis(
        fig[1, 1],
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ylims!(ax1, 1e-5, 1e5)
    ax2 = Axis(
        fig[1, 2],
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Ang. Mom.)",
    )
    ylims!(ax2, 1e-5, 1e5)

    # 2.2. Second row: rigid body
    ax3 = Axis(
        fig[2, 1],
        xticks = 0:400:800,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ylims!(ax3, 1e-5, 1e5)
    ax4 = Axis(
        fig[2, 2],
        xticks = 0:400:800,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Casimir)",
    )
    ylims!(ax4, 1e-15, 1e5)

    # 3. Plot the data
    # 3.1. First row: two body problem
    times_single,
    ground_truth_traj,
    predicted_traj_snde,
    predicted_traj_node,
    predicted_traj_anode,
    predicted_traj_sanode,
    times_multi,
    u_rel_err_snde,
    u_rel_err_node,
    u_rel_err_anode,
    u_rel_err_sanode,
    L_rel_err_snde,
    L_rel_err_node,
    L_rel_err_anode,
    L_rel_err_sanode = two_body_data

    _, u_rel_err_hnn, L_rel_err_hnn = hnn_data

    # Averages
    mean_rel_err_snde = vec(mean(u_rel_err_snde, dims = 1))
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_anode = vec(mean(u_rel_err_anode, dims = 1))
    mean_rel_err_sanode = vec(mean(u_rel_err_sanode, dims = 1))
    mean_rel_err_hnn = vec(mean(u_rel_err_hnn, dims = 1))
    linewidth = 1
    lines!(ax1, times_multi, mean_rel_err_snde; color = snde_colour, linewidth)
    lines!(ax1, times_multi, mean_rel_err_node; color = node_colour, linewidth)
    lines!(ax1, times_multi, mean_rel_err_anode; color = anode_colour, linewidth)
    lines!(ax1, times_multi, mean_rel_err_sanode; color = sanode_colour, linewidth)
    lines!(ax1, times_multi, mean_rel_err_hnn[1:10:end]; color = hnn_colour, linewidth)

    # Averages
    linewidth = 1
    drift_snde = vec(mean(L_rel_err_snde, dims = 1))
    drift_node = vec(mean(L_rel_err_node, dims = 1))
    drift_anode = vec(mean(L_rel_err_anode, dims = 1))
    drift_sanode = vec(mean(L_rel_err_sanode, dims = 1))
    drift_hnn = vec(mean(L_rel_err_hnn, dims = 1))
    lines!(ax2, times_multi, drift_snde; color = snde_colour, linewidth)
    lines!(ax2, times_multi, drift_node; color = node_colour, linewidth)
    lines!(ax2, times_multi, drift_anode; color = anode_colour, linewidth)
    lines!(ax2, times_multi, drift_sanode; color = sanode_colour, linewidth)
    lines!(ax2, times_multi, drift_hnn[1:10:end]; color = hnn_colour, linewidth)

    # 3.2. Second row: rigid body
    times_single,
    ground_truth_traj,
    predicted_traj_snde,
    predicted_traj_node,
    predicted_traj_anode,
    predicted_traj_sanode,
    times_multi,
    u_rel_err_snde,
    u_rel_err_node,
    u_rel_err_anode,
    u_rel_err_sanode,
    C_rel_err_snde,
    C_rel_err_node,
    C_rel_err_anode,
    C_rel_err_sanode = rigid_body_data

    _, u_rel_err_tpo, C_rel_err_tpo = tpo_data
    u_rel_err_tpo = u_rel_err_tpo[:, 1:10:end]
    C_rel_err_tpo = C_rel_err_tpo[:, 1:10:end]

    # Error in state
    r = 1:8000
    linewidth = 1
    mean_rel_err_snde = vec(mean(u_rel_err_snde, dims = 1))
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_anode = vec(mean(u_rel_err_anode, dims = 1))
    mean_rel_err_sanode = vec(mean(u_rel_err_sanode, dims = 1))
    mean_rel_err_tpo = vec(mean(u_rel_err_tpo, dims = 1))
    lines!(ax3, times_multi[r], mean_rel_err_snde[r]; color = snde_colour, linewidth)
    lines!(ax3, times_multi[r], mean_rel_err_node[r]; color = node_colour, linewidth)
    lines!(ax3, times_multi[r], mean_rel_err_anode[r]; color = anode_colour, linewidth)
    lines!(ax3, times_multi[r], mean_rel_err_sanode[r]; color = sanode_colour, linewidth)
    lines!(ax3, times_multi[r], mean_rel_err_tpo[r]; color = tpo_colour, linewidth)

    # Error in constraint
    linewidth = 1
    drift_snde = vec(mean(C_rel_err_snde, dims = 1))
    drift_node = vec(mean(C_rel_err_node, dims = 1))
    drift_anode = vec(mean(C_rel_err_anode, dims = 1))
    drift_sanode = vec(mean(C_rel_err_sanode, dims = 1))
    drift_tpo = vec(mean(C_rel_err_tpo, dims = 1))
    lines!(ax4, times_multi[r], drift_snde[r]; color = snde_colour, linewidth)
    lines!(ax4, times_multi[r], drift_node[r]; color = node_colour, linewidth)
    lines!(ax4, times_multi[r], drift_anode[r]; color = anode_colour, linewidth)
    lines!(ax4, times_multi[r], drift_sanode[r]; color = sanode_colour, linewidth)
    lines!(ax4, times_multi[r], drift_tpo[r]; color = tpo_colour, linewidth)

    # 4. Add the legend
    linewidth = 2
    elem_snde = [LineElement(color = snde_colour, linewidth = linewidth)]
    elem_node = [LineElement(color = node_colour, linewidth = linewidth)]
    elem_anode = [LineElement(color = anode_colour, linewidth = linewidth)]
    elem_sanode = [LineElement(color = sanode_colour, linewidth = linewidth)]
    elem_hnn = [LineElement(color = hnn_colour, linewidth = linewidth)]
    elem_tpo = [LineElement(color = tpo_colour, linewidth = linewidth)]
    Legend(
        fig[0, 1:2],
        [elem_node, elem_anode, elem_snde, elem_sanode, elem_hnn, elem_tpo],
        ["NODE", "ANODE", "SNODE", "SANODE", "HNN", "TPO"],
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        padding = 3,
        margin = (0, 0, -10, 0),
    )

    Label(fig[1, 1, Top()], "(a)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 2, Top()], "(b)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[2, 1, Top()], "(c)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[2, 2, Top()], "(d)", font = :bold, padding = (0, 3, 4, 0))
    fig
end

CairoMakie.save("figures/figure_7.pdf", fig)
