begin
    using JLD2
    using CairoMakie
    using Statistics
    using FileIO

    # Get the data
    data_path = "figures/data/robot_arm.jld2"
    times_single,
    ground_truth_traj,
    predicted_traj_snode,
    predicted_traj_node,
    times_multi,
    e_rel_err_snode,
    e_rel_err_node = JLD2.load_object(data_path)

    # Set up the figure
    size_inches = 1.2 .* (6, 2.5)
    size_pt = 72 .* size_inches
    linewidth = 1
    fig = Figure(resolution = size_pt, fontsize = 11, figure_padding = 6)
    ax1 = Axis(fig[1, 1], aspect = DataAspect())
    hidedecorations!(ax1)
    hidespines!(ax1)
    ax2 = Axis(
        fig[1, 2],
        xlabel = "Time (seconds)",
        xticks = 100:2:104,
        ylabel = L"e(\theta)",
    )
    xlims!(ax2, 99.7, 104.3)
    ax3 = Axis(
        fig[1, 3],
        yscale = log10,
        xlabel = "Time (seconds)",
        xticks = 0:250:500,
        ylabel = "Relative Error (Endpoint)",
    )

    # Plot schematic
    img = load("figures/robot_arm_schematic.pdf")
    image!(ax1, rotr90(img))

    # Plot single trajectory
    x = 9900
    r = x:(x+600)
    lines!(ax2, times_single[r], ground_truth_traj[1, r]; color = :black, linewidth = 2)
    lines!(ax2, times_single[r], predicted_traj_node[1, r]; color = :dodgerblue, linewidth)
    lines!(ax2, times_single[r], predicted_traj_snode[1, r]; color = :orangered, linewidth)

    # Plot relative error in the state
    linewidth = 0.1
    mean_rel_err_snode = vec(mean(e_rel_err_snode, dims = 1))
    mean_rel_err_node = vec(mean(e_rel_err_node, dims = 1))
    lines!(ax3, times_multi, mean_rel_err_snode; color = :orangered, linewidth)
    lines!(ax3, times_multi, mean_rel_err_node; color = :dodgerblue, linewidth)

    # Plot confidence intervals
    N = size(e_rel_err_snode, 1)
    error_snode = 1.96 * vec(std(e_rel_err_snode, dims = 1)) / sqrt(N)
    error_node = 1.96 * vec(std(e_rel_err_node, dims = 1)) / sqrt(N)
    lower_snode = mean_rel_err_snode .- error_snode
    lower_snode = [max(x, 0.00001) for x in lower_snode]  # tmp
    upper_snode = mean_rel_err_snode .+ error_snode
    lower_node = mean_rel_err_node .- error_node
    lower_node = [max(x, 0.00001) for x in lower_node]  # tmp
    upper_node = mean_rel_err_node .+ error_node

    band!(ax3, times_multi, lower_snode, upper_snode, color = (:orangered, 0.3))
    band!(ax3, times_multi, lower_node, upper_node, color = (:dodgerblue, 0.3))

    # Legend and labels
    linewidth = 2
    elem_truth = [LineElement(color = :black, linewidth = linewidth)]
    elem_node = [LineElement(color = :dodgerblue, linewidth = linewidth)]
    elem_snode = [LineElement(color = :orangered, linewidth = linewidth)]
    Legend(
        fig[2, 1:3],
        [elem_truth, elem_node, elem_snode],
        ["Ground Truth", "NODE", "SNODE"],
        orientation = :horizontal,
        padding = 3,
        margin = (0, 0, 0, -15),
    )
    Label(fig[1, 1, Top()], "(a)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 2, Top()], "(b)", font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 3, Top()], "(c)", font = :bold, padding = (0, 3, 4, 0))

    colgap!(fig.layout, 1, 7)
    fig
end

CairoMakie.save("figures/figure_4.pdf", fig)
