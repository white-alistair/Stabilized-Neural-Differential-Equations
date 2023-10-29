struct TwoBodyProblem{T} <: AbstractDynamicalSystem{T} end

# Ground truth equations of motion
function (system::TwoBodyProblem)(du, u, p, t)
    q₁, q₂, p₁, p₂ = u

    du[1] = p₁
    du[2] = p₂
    du[3] = -q₁ / (q₁^2 + q₂^2)^(3 / 2)
    du[4] = -q₂ / (q₁^2 + q₂^2)^(3 / 2)

    return nothing
end

function hamiltonian(u::AbstractVector{T}) where {T}
    q₁, q₂, p₁, p₂ = u
    return T(0.5) * (p₁^2 + p₂^2) - 1 / sqrt(q₁^2 + q₂^2)
end

function initial_conditions(::TwoBodyProblem{T}) where {T}
    e = 0.5 + 0.2 * rand()  # eccentricity drawn from [0.5, 0.7]
    return T[1-e, 0, 0, sqrt((1 + e) / (1 - e))]
end

function get_trajectories(
    system::TwoBodyProblem{T},
    experiment_version,
    seconds,
    dt,
    transient_seconds,
    solver,
    reltol,
    abstol,
    N,
    steps,
    stabilization_param,
    θ,
    restructure,
    augment_dim,
) where {T}
    systemBF = TwoBodyProblem{BigFloat}()
    trajectories = []
    for _ = 1:N
        u0 = initial_conditions(systemBF)
        H₀ = hamiltonian(u0)
        period = 2π * (2 * abs(H₀))^(-3 / 2)  # We want one full period for each trajectory
        time_series = integrate_trajectory(
            systemBF;
            seconds = period,
            dt,
            transient_seconds,
            solver,
            reltol,
            abstol,
            u0,
            NF = T,
            augment_dim,
        )
        u0 = time_series.trajectory[:, 1]
        t0 = time_series.times[1]

        γ = T(stabilization_param)
        if γ == 0
            rhs = @closure (u, θ, t) ->
                rhs_neural(u, θ, t, restructure, system, Val(experiment_version))
        else
            rhs = StabilizedNDE(
                u0,
                t0,
                γ,
                restructure,
                augment_dim,
                system,
                experiment_version,
            )
        end

        prob =
            ODEProblem{false,SciMLBase.FullSpecialize}(rhs, zeros(T), (zero(T), one(T)), θ)
        data_ms = multiple_shooting(prob, time_series; steps)
        push!(trajectories, data_ms)
    end
    return vcat(trajectories...)
end

function constraints(u, t, ::TwoBodyProblem)
    q₁, q₂, p₁, p₂ = u
    return [q₁ * p₂ - q₂ * p₁]
end

function constraints_jacobian(u, t, ::TwoBodyProblem)
    q₁, q₂, p₁, p₂ = u
    return [p₂ -p₁ -q₂ q₁]
end

# EXPERIMENT 1: Second order neural ODE
function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::TwoBodyProblem{T},
    ::Val{1},
) where {T}
    return get_mlp(4 => 2, hidden_layers, hidden_width, augment_dim, activation, T)
end

function rhs_neural(u, θ, t, re::Optimisers.Restructure, ::TwoBodyProblem, ::Val{1})
    q₁, q₂, p₁, p₂ = u
    return vcat(p₁, p₂, re(θ)(u))
end

# EXPERIMENT 2: HNN
function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::TwoBodyProblem{T},
    ::Val{2},
) where {T}
    mlp = Chain(
        x -> x.^2,  # Square inputs
        Dense(4 => hidden_width, activation),                                              # First hidden layer
        [Dense(hidden_width => hidden_width, activation) for _ = 1:(hidden_layers-1)]...,  # Remaining hidden layers
        Dense(hidden_width => 1),                                                          # Output layer
    )
    return destructure(Flux._paramtype(T, mlp))
end

function rhs_neural(u, θ, t, re::Optimisers.Restructure, ::TwoBodyProblem, ::Val{2})
    ∇H = only(Zygote.gradient(x -> sum(re(θ)(x)), u))
    J = [0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
    return J * ∇H
end
