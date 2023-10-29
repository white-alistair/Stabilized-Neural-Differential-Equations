begin
    using JLD2
    using CairoMakie
    using Statistics

    data_path = "figures/data/double_pendulum.jld2"

    times_short,
    u_rel_err_snode,
    u_rel_err_node,
    E_rel_err_snode,
    E_rel_err_node,
    times_long,
    E_rel_err_snode_long,
    E_rel_err_node_long,
    distances = JLD2.load_object(data_path)

    size_inches = 1.2 .* (6, 1.75)
    size_pt = 72 .* size_inches
    linewidth = 1
    fig = Figure(resolution = size_pt, fontsize = 11, figure_padding = 3)

    ax1 = Axis(
        fig[1, 1],
        xticks = 0:2.5:5,
        yscale = log10,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (State)",
    )
    ax2 = Axis(
        fig[1, 2],
        yscale = log10,
        xticks = 0:250:500,
        xlabel = "Time (seconds)",
        ylabel = "Relative Error (Energy)",
    )
    ax3 = Axis(
        fig[1, 3],
        xlabel = L"\gamma",
        ylabel = "Hellinger Distance",
        xticks = (1:5, ["0", "1", "2", "4", "8"]),
    )

    # Plot relative error in state
    linewidth = 1
    mean_rel_err_node = vec(mean(u_rel_err_node, dims = 1))
    mean_rel_err_snode = vec(mean(u_rel_err_snode, dims = 1))
    lines!(ax1, times_short, mean_rel_err_node; color = :dodgerblue, linewidth)
    lines!(ax1, times_short, mean_rel_err_snode; color = :orangered, linewidth)

    N = size(u_rel_err_snode, 1)
    error_snode = 1.96 * vec(std(u_rel_err_snode, dims = 1)) / sqrt(N)
    error_node = 1.96 * vec(std(u_rel_err_node, dims = 1)) / sqrt(N)
    lower_snode = mean_rel_err_snode .- error_snode
    lower_snode = [max(x, 0.00001) for x in lower_snode]  # tmp
    upper_snode = mean_rel_err_snode .+ error_snode
    lower_node = mean_rel_err_node .- error_node
    upper_node = mean_rel_err_node .+ error_node

    band!(ax1, times_short, lower_snode, upper_snode, color = (:orangered, 0.3))
    band!(ax1, times_short, lower_node, upper_node, color = (:dodgerblue, 0.3))

    # Plot relative error in energy (long)
    linewidth = 0.2
    mean_rel_err_node = vec(mean(E_rel_err_node_long[1:5, :], dims = 1))
    mean_rel_err_snode = vec(mean(E_rel_err_snode_long[1:5, :], dims = 1))
    lines!(ax2, times_long, mean_rel_err_node; color = :dodgerblue, linewidth)
    lines!(ax2, times_long, mean_rel_err_snode; color = :orangered, linewidth)

    # Plot Hellinger distance
    avg_distance = mapslices(mean, distances, dims = 2)
    std_error = mapslices(x -> std(x) / sqrt(length(x)), distances, dims = 2)
    scatter!(
        ax3,
        1:length(avg_distance),
        vec(avg_distance),
        marker = :circle,
        color = :dodgerblue,
        markersize = 8,
    )
    errorbars!(
        ax3,
        1:length(avg_distance),
        vec(avg_distance),
        vec(1.96 .* std_error),
        linewidth = 1,
    )

    # Add legend and labels
    linewidth = 2
    elem_node = [LineElement(color = :dodgerblue, linewidth = linewidth)]
    elem_snode = [LineElement(color = :orangered, linewidth = linewidth)]
    Legend(
        fig[1, 0],
        [elem_node, elem_snode],
        ["NODE", "SNODE"],
        orientation = :vertical,
        tellwidth = true,
        tellheight = false,
        padding = 3,
        margin = (0, -10, 0, 0),
        labelsize = 8,
    )

    Label(fig[1, 1, Top()], "(a)", fontsize = 9, font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 2, Top()], "(b)", fontsize = 9, font = :bold, padding = (0, 3, 4, 0))
    Label(fig[1, 3, Top()], "(c)", fontsize = 9, font = :bold, padding = (0, 3, 4, 0))
    fig
end

CairoMakie.save("figures/figure_5.pdf", fig)
