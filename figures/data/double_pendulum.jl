using StabilizedNDEs:
    DoublePendulum,
    rhs_neural,
    constraints,
    constraints_jacobian,
    initial_conditions,
    StabilizedNDE,
    get_relative_error,
    deserialize,
    get_mlp
using OrdinaryDiffEq
using Random
using FastClosures
using JLD2

system = DoublePendulum{Float64}()
exp_version = 1
tol = 1e-6

# NODE
id_node = 26657980
θ_node, re_node, _ = deserialize(id_node)

# SNODE
id_snode = 26657985
θ_snode, re_snode, γ = deserialize(id_snode)

# Multiple short trajectories
begin
    Random.seed!(2)

    t0 = 0.0
    T = 5
    dt = 0.01
    times_short = collect(dt:dt:T)

    N = 300
    u_rel_err_node = zeros(N, length(times_short))
    u_rel_err_snode = zeros(N, length(times_short))
    E_rel_err_node = zeros(N, length(times_short))
    E_rel_err_snode = zeros(N, length(times_short))

    Threads.@threads for i = 1:N
        @info i
        flush(stderr)
        u0 = initial_conditions(system)

        # Constraints (always the same)
        g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)
        E0 = abs(constraints(u0, t0, system)[1])

        # Ground truth
        prob = ODEProblem(system, u0, (t0, T))
        sol = solve(prob, Tsit5(); saveat = times_short, abstol = tol, reltol = tol)
        ground_truth = Array(sol)

        # NODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
        prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
        sol = solve(prob, Tsit5(); saveat = times_short, abstol = tol, reltol = tol)
        predicted = Array(sol)
        E_rel_err_node[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / E0, predicted, dims = 1))
        u_rel_err_node[i, :] .= get_relative_error(predicted, ground_truth)

        # SNODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            (J' * inv(J * J'))
        end
        snode = StabilizedNDE(f, γ, F, g)
        prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
        sol = solve(prob, Tsit5(); saveat = times_short, abstol = tol, reltol = tol)
        predicted = Array(sol)
        E0 = abs(constraints(u0, t0, system)[1])
        E_rel_err_snode[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / E0, predicted, dims = 1))
        u_rel_err_snode[i, :] .= get_relative_error(predicted, ground_truth)

        GC.gc()
        ccall(:malloc_trim, Cvoid, (Cint,), 0)
    end
end

# Some long trajectories
begin
    Random.seed!(2)

    t0 = 0.0
    T = 500
    dt = 0.1
    times_long = collect(dt:dt:T)

    N = 100
    E_rel_err_node_long = zeros(N, length(times_long))
    E_rel_err_snode_long = zeros(N, length(times_long))

    Threads.@threads for i = 1:N
        @info i
        flush(stderr)
        u0 = initial_conditions(system)

        # Constraints (always the same)
        g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)
        E0 = abs(constraints(u0, t0, system)[1])

        # Ground truth
        prob = ODEProblem(system, u0, (t0, T))
        sol = solve(prob, Tsit5(); saveat = times_long, abstol = tol, reltol = tol)
        ground_truth = Array(sol)

        # NODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
        prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
        sol = solve(prob, Tsit5(); saveat = times_long, abstol = tol, reltol = tol)
        predicted = Array(sol)
        E_rel_err_node_long[i, 1:size(predicted)[2]] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / E0, predicted, dims = 1))

        # SNODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            (J' * inv(J * J'))
        end
        snode = StabilizedNDE(f, γ, F, g)
        prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
        sol = solve(prob, Tsit5(); saveat = times_long, abstol = tol, reltol = tol)
        predicted = Array(sol)
        E0 = abs(constraints(u0, t0, system)[1])
        E_rel_err_snode_long[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / E0, predicted, dims = 1))

        GC.gc()
        ccall(:malloc_trim, Cvoid, (Cint,), 0)
    end
end

data_path = "figures/data/double_pendulum.jld2"
JLD2.save_object(
    data_path,
    (
        times_short,
        u_rel_err_node,
        u_rel_err_snode,
        E_rel_err_node,
        E_rel_err_snode,
        times_long,
        E_rel_err_node_long,
        E_rel_err_snode_long,
    ),
)
