using StabilizedNDEs:
    RobotArm,
    pose,
    endpoint,
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

system = RobotArm{Float64}()
exp_version = 1
tol = 1e-9

# NODE
id_node = 26657925
θ_node, re_node, _ = deserialize(id_node)

# SNODE
id_snode = 26657930
θ_snode, re_snode, γ = deserialize(id_snode)

# Single trajectory
begin
    Random.seed!(2)
    u0 = initial_conditions(system)

    t0 = 0.0
    T = 500.0
    dt = 0.01
    times_single = collect(dt:dt:T)

    # Constraints (always the same)
    g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)

    # Ground truth
    prob = ODEProblem(system, u0, (t0, T))
    sol = solve(prob, Vern9(); saveat = times_single, abstol = 1e-12, reltol = 1e-12)
    ground_truth = Array(sol)

    # NODE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
    prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_node = Array(sol)

    # SNODE
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
    F = @closure (u, t) -> begin
        J = constraints_jacobian(u, t, system)
        J' * inv(J * J')
    end
    snode = StabilizedNDE(f, γ, F, g)
    prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
    sol = solve(prob, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_snode = Array(sol)

    # SNODE (fast)
    f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
    F = @closure (u, t) -> begin
        J = constraints_jacobian(u, t, system)
        J'
    end
    snode = StabilizedNDE(f, γ, F, g)
    prob_fast = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
    sol = solve(prob_fast, Tsit5(); saveat = times_single, abstol = tol, reltol = tol)
    predicted_traj_snode_fast = Array(sol)
end

# 2. Multiple trajectories
begin
    Random.seed!(2)

    t0 = 0.0
    T = 500.0
    dt = 0.1
    times_multi = collect(dt:dt:T)

    N = 10
    e_rel_err_node = zeros(N, length(times_multi))
    e_rel_err_snode = zeros(N, length(times_multi))
    e_rel_err_snode_fast = zeros(N, length(times_multi))

    Threads.@threads for i = 1:N
        @info i
        flush(stderr)
        u0 = initial_conditions(system)

        g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)
        e0 = abs(constraints(u0, t0, system)[1])

        # Ground truth
        prob = ODEProblem(system, u0, (t0, T))
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        ground_truth = Array(sol)
        ground_truth_e = mapslices(endpoint, ground_truth, dims = 1)

        # NODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_node, system, Val(exp_version))
        prob = ODEProblem{false}(f, u0, (t0, T), θ_node)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted = Array(sol)
        predicted_node_e = mapslices(endpoint, predicted, dims = 1)
        e_rel_err_node[i, :] .= get_relative_error(predicted_node_e, ground_truth_e)

        # SNODE
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            J' * inv(J * J')
        end
        snode = StabilizedNDE(f, γ, F, g)
        prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted = Array(sol)
        predicted_snode_e = mapslices(endpoint, predicted, dims = 1)
        e_rel_err_snode[i, :] .= get_relative_error(predicted_snode_e, ground_truth_e)

        # SNODE (fast)
        f = @closure (u, θ, t) -> rhs_neural(u, θ, t, re_snode, system, Val(exp_version))
        F = @closure (u, t) -> begin
            J = constraints_jacobian(u, t, system)
            J'
        end
        snode = StabilizedNDE(f, γ, F, g)
        prob = ODEProblem{false}(snode, u0, (t0, T), θ_snode)
        sol = solve(prob, Tsit5(); saveat = times_multi, abstol = tol, reltol = tol)
        predicted = Array(sol)
        predicted_snode_e = mapslices(endpoint, predicted, dims = 1)
        e_rel_err_snode_fast[i, :] .= get_relative_error(predicted_snode_e, ground_truth_e)
    end
end
