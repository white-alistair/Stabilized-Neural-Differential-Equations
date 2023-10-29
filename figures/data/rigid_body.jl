using StabilizedNDEs:
    RigidBody,
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

system = RigidBody{Float64}()
exp_version = 1
tol = 1e-9

# NODE
id_node = 26643906
θ_node, re_node, _ = deserialize(id_node)

# SNDE
id_snde = 26643912
θ_snde, re_snde, γ = deserialize(id_snde)

# ANODE
id_anode = 27153184
θ_anode, re_anode, _ = deserialize(id_anode)

# # SANODE
id_sanode = 27162880
θ_sanode, re_sanode, _ = deserialize(id_sanode)

# Single trajectory
begin
    Random.seed!(19)
    u0 = initial_conditions(system)

    t0 = 0.0
    T = 1000.0
    dt = 0.01
    times_single = collect(dt:dt:T)

    # Constraints (always the same)
    g = @closure (u, t) ->
        constraints(u, t, system, Val(exp_version)) .-
        constraints(u0, t0, system, Val(exp_version))

    # Ground truth
    prob = ODEProblem(system, u0, (t0, T))
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    ground_truth_traj = Array(sol)

    # NODE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
    prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_node = Array(sol)

    # SNDE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snde, system, Val(exp_version))
    F = @closure (u, t) -> (J = constraints_jacobian(u, t, system, Val(exp_version));
    (J' * inv(J * J')))
    snde = StabilizedNDE(f, g, F, γ)
    prob = ODEProblem{false}(snde, u0, (t0, T), θ_snde)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_snde = Array(sol)

    # ANODE
    augment_dim = 2
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_anode, system, Val(exp_version))
    prob = ODEProblem{false}(f, vcat(u0, zeros(augment_dim)), (t0, T), θ_anode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_anode = Array(sol)[1:end-augment_dim, :]

    # SANODE
    augment_dim = 2
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_sanode, system, Val(exp_version))
    F = @closure (u, t) -> (J = constraints_jacobian(u, t, system, Val(exp_version));
    vcat(J' * inv(J * J'), zeros(augment_dim)))
    snde = StabilizedNDE(f, g, F, γ)
    prob = ODEProblem{false}(snde, vcat(u0, zeros(augment_dim)), (t0, T), θ_sanode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_sanode = Array(sol)[1:end-augment_dim, :]
end

# 2. Multiple trajectories
begin
    Random.seed!(2)

    t0 = 0.0
    T = 1000.0
    dt = 0.1
    times_multi = collect(dt:dt:T)

    N = 100
    u_rel_err_snde = zeros(N, length(times_multi))
    u_rel_err_node = zeros(N, length(times_multi))
    u_rel_err_anode = zeros(N, length(times_multi))
    u_rel_err_sanode = zeros(N, length(times_multi))
    C_rel_err_snde = zeros(N, length(times_multi))
    C_rel_err_node = zeros(N, length(times_multi))
    C_rel_err_anode = zeros(N, length(times_multi))
    C_rel_err_sanode = zeros(N, length(times_multi))

    Threads.@threads for i = 1:N
        @info i
        flush(stderr)
        u0 = initial_conditions(system)

        # Constraints (always the same)
        g = @closure (u, t) ->
            constraints(u, t, system, Val(exp_version)) .-
            constraints(u0, t0, system, Val(exp_version))
        C0 = abs(constraints(u0, t0, system, Val(exp_version))[1])

        # Ground truth
        prob = ODEProblem(system, u0, (t0, T))
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        ground_truth = Array(sol)

        # NODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
        prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)
        C_rel_err_node[i, 1:size(predicted_traj)[2]] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / C0, predicted_traj, dims = 1))
        u_rel_err_node[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # SNDE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snde, system, Val(exp_version))
        F = @closure (u, t) -> (J = constraints_jacobian(u, t, system, Val(exp_version));
        (J' * inv(J * J')))
        snde = StabilizedNDE(f, g, F, γ)
        prob = ODEProblem{false}(snde, u0, (t0, T), θ_snde)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)
        C_rel_err_snde[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / C0, predicted_traj, dims = 1))
        u_rel_err_snde[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # ANODE
        augment_dim = 2
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_anode, system, Val(exp_version))
        prob = ODEProblem{false}(f, vcat(u0, zeros(augment_dim)), (t0, T), θ_anode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)[1:end-augment_dim, :]
        C_rel_err_anode[i, :] .=
            vec(mapslices(u -> abs(g(u, nothing)[1]) / C0, predicted_traj, dims = 1))
        u_rel_err_anode[i, :] .= get_relative_error(predicted_traj, ground_truth)

        # SANODE
        augment_dim = 2
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_sanode, system, Val(exp_version))
        F = @closure (u, t) -> (J = constraints_jacobian(u, t, system, Val(exp_version));
        vcat(J' * inv(J * J'), zeros(augment_dim)))
        snde = StabilizedNDE(f, g, F, γ)
        prob = ODEProblem{false}(snde, vcat(u0, zeros(augment_dim)), (t0, T), θ_sanode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted_traj = Array(sol)[1:end-augment_dim, :]
        C_rel_err_sanode[i, :] .=
            vec(mapslices(u -> abs(g(u, nGC.gc()
        ccall(:malloc_trim, Cvoid, (Cint,), 0)othing)[1]) / C0, predicted_traj, dims = 1))
        u_rel_err_sanode[i, :] .= get_relative_error(predicted_traj, ground_truth)

        
    end
end

data_path = "figures/data/rigid_body.jld2"
JLD2.save_object(
    data_path,
    (
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
        C_rel_err_sanode,
    ),
)
