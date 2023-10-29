begin
    using JLD2
    using CairoMakie
    using Statistics

    # 1. Get the data
    two_body_data = JLD2.load_object("figures/data/two_body_problem.jld2")
    rigid_body_data = JLD2.load_object("figures/data/rigid_body.jld2")
    converter_data = JLD2.load_object("figures/data/converter.jld2")

    # 2. Set up the figure
    size_inches = 1.6 .* (5, 5)
    size_pt = 72 .* size_inches
    linewidth = 1
    fig = Figure(resolution = size_pt, fontsize = 12, figure_padding = 5)

    truth_colour = :black
    node_colour = :dodgerblue
    anode_colour = :darkgreen
    snode_colour = :orangered
    sanode_colour = :purple

    # 2.1. First row: two body problem
    ax1 = Axis(fig[1, 1], xlabel = L"x", ylabel = L"y", aspect = DataAspect())
    xlims!(ax1, -1.7, 0.7)
    ylims!(ax1, -1.0, 1.0)
    ax2 = Axis(
        fig[1, 2],
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ylims!(ax2, 1e-5, 1e5)
    ax3 = Axis(
        fig[1, 3],
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Ang. Mom.)",
    )
    ylims!(ax3, 1e-5, 1e5)

    # 2.2. Second row: rigid body
    ax4 = Axis3(
        fig[2, 1],
        xlabel = L"y_1",
        ylabel = L"y_2",
        zlabel = L"y_3",
        xticks = -0.2:0.2:0.2,
        yticks = -0.2:0.2:0.2,
        zticks = 0.97:0.02:1.01,
        xlabeloffset = 25,
        ylabeloffset = 25,
        zlabeloffset = 37,
        protrusions = 0,
    )
    ax5 = Axis(
        fig[2, 2],
        xticks = 0:400:800,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ylims!(ax5, 1e-5, 1e5)
    ax6 = Axis(
        fig[2, 3],
        xticks = 0:400:800,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Casimir)",
    )
    ylims!(ax6, 1e-5, 1e5)

    # 2.3. Third row: converter
    ax7 = Axis(fig[3, 1], xlabel = "Time (seconds)", ylabel = L"v_1\,[V]")
    ax8 = Axis(
        fig[3, 2],
        xticks = 0:40:80,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ylims!(ax8, 1e-5, 1e5)
    ax9 = Axis(
        fig[3, 3],
        xticks = 0:40:80,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Energy)",
    )
    ylims!(ax9, 1e-5, 1e5)

    # 3. Plot the data
    # 3.1. First row: two body problem
    times_single,
    ground_truth_traj,
    predicted_traj_snode,
    predicted_traj_node,
    predicted_traj_anode,
    predicted_traj_sanode,
    times_multi,
    u_rel_err_snode,
    u_rel_err_node,
    u_rel_err_anode,
    u_rel_err_sanode,
    L_rel_err_snode,
    L_rel_err_node,
    L_rel_err_anode,
    L_rel_err_sanode = two_body_data

    linewidth = 0.8
    r = 1:4500
    lines!(
        ax1,
        predicted_traj_node[1, r],
        predicted_traj_node[2, r];
        color = node_colour,
        linewidth,
    )
    lines!(
        ax1,
        predicted_traj_anode[1, :],
        predicted_traj_anode[2, :];
        color = anode_colour,
        linewidth,
    )
    lines!(
        ax1,
        ground_truth_traj[1, :],
        ground_truth_traj[2, :];
        color = truth_colour,
        linewidth = 1.2,
    )
    lines!(
        ax1,
        predicted_traj_snode[1, :],
        predicted_traj_snode[2, :];
        color = snode_colour,
        linewidth,
    )
    lines!(
        ax1,
        predicted_traj_sanode[1, :],
        predicted_traj_sanode[2, :];
        color = sanode_colour,
        linewidth,
    )

    # Averages
    mean_rel_err_snode = vec(mean(u_rel_err_snode, dims = 1))
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_anode = vec(mean(u_rel_err_anode, dims = 1))
    mean_rel_err_sanode = vec(mean(u_rel_err_sanode, dims = 1))
    linewidth = 1
    lines!(ax2, times_multi, mean_rel_err_snode; color = snode_colour, linewidth)
    lines!(ax2, times_multi, mean_rel_err_node; color = node_colour, linewidth)
    lines!(ax2, times_multi, mean_rel_err_anode; color = anode_colour, linewidth)
    lines!(ax2, times_multi, mean_rel_err_sanode; color = sanode_colour, linewidth)

    # Averages
    linewidth = 1
    drift_snde = vec(mean(L_rel_err_snode, dims = 1))
    drift_node = vec(mean(L_rel_err_node, dims = 1))
    drift_anode = vec(mean(L_rel_err_anode, dims = 1))
    drift_sanode = vec(mean(L_rel_err_sanode, dims = 1))
    lines!(ax3, times_multi, drift_snde; color = snode_colour, linewidth)
    lines!(ax3, times_multi, drift_node; color = node_colour, linewidth)
    lines!(ax3, times_multi, drift_anode; color = anode_colour, linewidth)
    lines!(ax3, times_multi, drift_sanode; color = sanode_colour, linewidth)

    # 3.2. Second row: rigid body
    times_single,
    ground_truth_traj,
    predicted_traj_snode,
    predicted_traj_node,
    predicted_traj_anode,
    predicted_traj_sanode,
    times_multi,
    u_rel_err_snode,
    u_rel_err_node,
    u_rel_err_anode,
    u_rel_err_sanode,
    C_rel_err_snode,
    C_rel_err_node,
    C_rel_err_anode,
    C_rel_err_sanode = rigid_body_data

    linewidth = 0.1
    lines!(
        ax4,
        predicted_traj_node[1, :],
        predicted_traj_node[2, :],
        predicted_traj_node[3, :];
        color = node_colour,
        linewidth,
    )
    lines!(
        ax4,
        predicted_traj_anode[1, :],
        predicted_traj_anode[2, :],
        predicted_traj_anode[3, :];
        color = anode_colour,
        linewidth,
    )
    lines!(
        ax4,
        predicted_traj_sanode[1, :],
        predicted_traj_sanode[2, :],
        predicted_traj_sanode[3, :];
        color = sanode_colour,
        linewidth,
    )
    lines!(
        ax4,
        predicted_traj_snode[1, :],
        predicted_traj_snode[2, :],
        predicted_traj_snode[3, :];
        color = snode_colour,
        linewidth,
    )
    lines!(
        ax4,
        ground_truth_traj[1, :],
        ground_truth_traj[2, :],
        ground_truth_traj[3, :];
        color = truth_colour,
        linewidth = 1.0,
    )

    # Averages
    r = 1:8000
    linewidth = 1
    mean_rel_err_snode = vec(mean(u_rel_err_snode, dims = 1))
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_anode = vec(mean(u_rel_err_anode, dims = 1))
    mean_rel_err_sanode = vec(mean(u_rel_err_sanode, dims = 1))
    lines!(ax5, times_multi[r], mean_rel_err_snode[r]; color = snode_colour, linewidth)
    lines!(ax5, times_multi[r], mean_rel_err_node[r]; color = node_colour, linewidth)
    lines!(ax5, times_multi[r], mean_rel_err_anode[r]; color = anode_colour, linewidth)
    lines!(ax5, times_multi[r], mean_rel_err_sanode[r]; color = sanode_colour, linewidth)

    # Averages
    linewidth = 1
    drift_snode = vec(mean(C_rel_err_snode, dims = 1))
    drift_node = vec(mean(C_rel_err_node, dims = 1))
    drift_anode = vec(mean(C_rel_err_anode, dims = 1))
    drift_sanode = vec(mean(C_rel_err_sanode, dims = 1))
    lines!(ax6, times_multi[r], drift_snode[r]; color = snode_colour, linewidth)
    lines!(ax6, times_multi[r], drift_node[r]; color = node_colour, linewidth)
    lines!(ax6, times_multi[r], drift_anode[r]; color = anode_colour, linewidth)
    lines!(ax6, times_multi[r], drift_sanode[r]; color = sanode_colour, linewidth)

    # 3.3. Third row: converter
    times_single,
    ground_truth_traj,
    predicted_traj_snode,
    predicted_traj_node,
    predicted_traj_anode,
    predicted_traj_sanode,
    times_multi,
    u_rel_err_snode,
    u_rel_err_node,
    u_rel_err_anode,
    u_rel_err_sanode,
    E_rel_err_snode,
    E_rel_err_node,
    E_rel_err_anode,
    E_rel_err_sanode = converter_data

    # Single trajectory
    linewidth = 0.5
    lines!(ax7, times_single, ground_truth_traj[1, :]; color = truth_colour, linewidth)
    lines!(ax7, times_single, predicted_traj_node[1, :]; color = node_colour, linewidth)
    lines!(ax7, times_single, predicted_traj_snode[1, :]; color = snode_colour, linewidth)
    lines!(ax7, times_single, predicted_traj_anode[1, :]; color = anode_colour, linewidth)
    lines!(ax7, times_single, predicted_traj_sanode[1, :]; color = sanode_colour, linewidth)

    # Relative error in the state
    linewidth = 1
    mean_rel_err_snode = vec(mean(u_rel_err_snode, dims = 1))
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_anode = vec(mean(u_rel_err_anode, dims = 1))
    mean_rel_err_sanode = vec(mean(u_rel_err_sanode, dims = 1))
    lines!(ax8, times_multi, mean_rel_err_snode; color = snode_colour, linewidth)
    lines!(ax8, times_multi, mean_rel_err_node; color = node_colour, linewidth)
    lines!(ax8, times_multi, mean_rel_err_anode; color = anode_colour, linewidth)
    lines!(ax8, times_multi, mean_rel_err_sanode; color = sanode_colour, linewidth)

    # Relative error in the constraint
    drift_snode = vec(mean(E_rel_err_snode, dims = 1))
    drift_node = vec(mean(E_rel_err_node, dims = 1))
    drift_anode = vec(mean(E_rel_err_anode, dims = 1))
    drift_sanode = vec(mean(E_rel_err_sanode, dims = 1))

    r = 1:800
    lines!(ax9, times_multi[r], drift_snode[r]; color = snode_colour, linewidth)
    lines!(ax9, times_multi[r], drift_node[r]; color = node_colour, linewidth)
    lines!(ax9, times_multi[r], drift_anode[r]; color = anode_colour, linewidth)
    lines!(ax9, times_multi[r], drift_sanode[r]; color = sanode_colour, linewidth)

    # 4. Add the legend
    linewidth = 2
    elem_truth = [LineElement(color = truth_colour, linewidth = linewidth)]
    elem_snde = [LineElement(color = snode_colour, linewidth = linewidth)]
    elem_node = [LineElement(color = node_colour, linewidth = linewidth)]
    elem_anode = [LineElement(color = anode_colour, linewidth = linewidth)]
    elem_sanode = [LineElement(color = sanode_colour, linewidth = linewidth)]
    Legend(
        fig[0, 1:3],
        [elem_truth, elem_node, elem_anode, elem_snde, elem_sanode],
        ["Ground Truth", "NODE", "ANODE", "SNODE", "SANODE"],
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        padding = 3,
        margin = (0, 0, -10, 0),
    )

    Label(fig[1, 1, Top()], "(a)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 2, Top()], "(b)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 3, Top()], "(c)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[2, 1, Top()], "(d)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[2, 2, Top()], "(e)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[2, 3, Top()], "(f)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[3, 1, Top()], "(g)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[3, 2, Top()], "(h)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[3, 3, Top()], "(i)", font = :bold, padding = (0, 3, 4, 0))
    fig
end

CairoMakie.save("figures/figure_2.pdf", fig)
