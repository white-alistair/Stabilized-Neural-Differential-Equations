using StabilizedNDEs:
    TwoBodyProblem,
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

system = TwoBodyProblem{Float64}()
exp_version = 1
tol = 1e-9

# NODE
id_node = 26643818
θ_node, re_node, _ = deserialize(id_node)

# SNODE
id_snode = 26643822
θ_snode, re_snode, γ_snode = deserialize(id_snode)

# ANODE
id_anode = 27148810
θ_anode, re_anode, _ = deserialize(id_anode)

# SANODE
id_sanode = 27162862
θ_sanode, re_sanode, γ_sanode = deserialize(id_sanode)

# Single trajectory
begin
    Random.seed!(2)

    u0 = initial_conditions(system)

    t0 = 0.0
    T = 60.0
    dt = 0.01
    times_single = collect(dt:dt:T)

    # Constraints (always the same)
    g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)

    # Ground truth
    prob = ODEProblem(system, u0, (t0, T))
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    ground_truth_traj = Array(sol)

    # NODE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
    prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_node = Array(sol)

    # SNODE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
    F = @closure (u, t) -> begin
        J = constraints_jacobian(u, t, system)
        (J' * inv(J * J'))
    end
    snode = StabilizedNDE(f, γ_snode, F, g)
    prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_snode = Array(sol)

    # ANODE
    augment_dim = 2
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_anode, system, Val(exp_version))
    prob = ODEProblem{false}(f, vcat(u0, zeros(augment_dim)), (t0, T), θ_anode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_anode = Array(sol)[1:end-augment_dim, :]

    # SANODE
    augment_dim = 2
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_sanode, system, Val(exp_version))
    F = @closure (u, t) -> begin
        J = constraints_jacobian(u, t, system)
        vcat(J' * inv(J * J'), zeros(augment_dim))
    end
    sanode = StabilizedNDE(f, γ_sanode, F, g)
    prob = ODEProblem{false}(sanode, vcat(u0, zeros(augment_dim)), (t0, T), θ_sanode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_sanode = Array(sol)[1:end-augment_dim, :]
end

# 2. Multiple trajectories
begin
    Random.seed!(2)

    t0 = 0.0
    T = 100.0
    dt = 0.1
    times_multi = collect(dt:dt:T)

    N = 100

    u_rel_err_node = zeros(N, length(times_multi))
    u_rel_err_snode = zeros(N, length(times_multi))
    u_rel_err_snde_fast = zeros(N, length(times_multi))
    u_rel_err_anode = zeros(N, length(times_multi))
    u_rel_err_sanode = zeros(N, length(times_multi))

    L_rel_err_node = zeros(N, length(times_multi))
    L_rel_err_snode = zeros(N, length(times_multi))
    L_rel_err_snde_fast = zeros(N, length(times_multi))
    L_rel_err_anode = zeros(N, length(times_multi))
    L_rel_err_sanode = zeros(N, length(times_multi))

    Threads.@threads for i = 1:N
        @info i
        flush(stderr)
        u0 = initial_conditions(system)

        # Constraints (always the same)
        g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)
        L0 = abs(constraints(u0, t0, system)[1])

        # Ground truth
        prob = ODEProblem(system, u0, (t0, T))
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        ground_truth = Array(sol)

        # NODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
        prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)
        L_rel_err_node[i, 1:size(predicted_traj)[2]] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / L0, predicted_traj, dims = 1))
        u_rel_err_node[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # SNDE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            (J' * inv(J * J'))
        end
        snde = StabilizedNDE(f, γ_snode, F, g)
        prob = ODEProblem{false}(snde, u0, (t0, T), θ_snode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)
        L_rel_err_snode[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / L0, predicted_traj, dims = 1))
        u_rel_err_snode[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # ANODE
        augment_dim = 2
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_anode, system, Val(exp_version))
        prob = ODEProblem{false}(f, vcat(u0, zeros(augment_dim)), (t0, T), θ_anode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)[1:end-augment_dim, :]
        L_rel_err_anode[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / L0, predicted_traj, dims = 1))
        u_rel_err_anode[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # SANODE
        augment_dim = 2
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_sanode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            vcat(J' * inv(J * J'), zeros(augment_dim))
        end
        snde = StabilizedNDE(f, γ_sanode, F, g)
        prob = ODEProblem{false}(snde, vcat(u0, zeros(augment_dim)), (t0, T), θ_sanode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)[1:end-augment_dim, :]
        L_rel_err_sanode[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / L0, predicted_traj, dims = 1))
        u_rel_err_sanode[i, :] .= get_relative_error(predicted_traj, ground_truth)

        GC.gc()
        ccall(:malloc_trim, Cvoid, (Cint,), 0)
    end
end

data_path = "figures/data/two_body_problem.jld2"
JLD2.save_object(
    data_path,
    (
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
        L_rel_err_sanode,
    ),
)
