"""
Given a system, integrate a trajectory and return the result as a TimeSeries object.
"""
function integrate_trajectory(
    system::AbstractDynamicalSystem{T};
    seconds::T,
    dt::T,
    transient_seconds::T = zero(T),
    solver = Vern9(),
    reltol = 1e-24,
    abstol = 1e-24,
    u0::AbstractVector{T},
    maxiters = typemax(Int),
    in_place = true,
    NF,
    augment_dim = nothing,
) where {T<:AbstractFloat}
    tspan = (zero(T), one(T))  # Arbitrary
    prob = ODEProblem{in_place}(system, u0, tspan; saveat = dt)

    if transient_seconds > 0
        transient_prob = remake(prob, tspan = (zero(T), transient_seconds))
        transient_data = solve(transient_prob, solver; reltol, abstol, maxiters)
        u0 = transient_data[end]
    else
        u0 = prob.u0
    end

    prob = remake(prob, u0 = u0, tspan = (zero(T), seconds))
    sol = solve(prob, solver; reltol, abstol, maxiters)
    trajectory = Array(sol)

    if !isnothing(augment_dim)
        trajectory = vcat(trajectory, zeros(T, augment_dim, size(trajectory)[2]))
    end

    return TimeSeries{NF}(sol.t, trajectory)
end

function get_data(
    system,
    experiment_version,
    T,
    dt,
    transient_seconds,
    solver,
    reltol,
    abstol,
    n_train,
    n_valid,
    n_test,
    steps,
    stabilization_param,
    θ,
    restructure,
    augment_dim,
)
    @info "Generating training data..."
    train_data = get_trajectories(
        system,
        experiment_version,
        T,
        dt,
        transient_seconds,
        solver,
        reltol,
        abstol,
        n_train,
        steps,
        stabilization_param,
        θ,
        restructure,
        augment_dim,
    )
    valid_data = get_trajectories(
        system,
        experiment_version,
        T,
        dt,
        transient_seconds,
        solver,
        reltol,
        abstol,
        n_valid,
        steps,
        stabilization_param,
        θ,
        restructure,
        augment_dim,
    )
    test_data = get_trajectories(
        system,
        experiment_version,
        T,
        dt,
        transient_seconds,
        solver,
        reltol,
        abstol,
        n_test,
        steps,
        stabilization_param,
        θ,
        restructure,
        augment_dim,
    )
    @info "Finished generating training data."
    return (; train_data, valid_data, test_data)
end
