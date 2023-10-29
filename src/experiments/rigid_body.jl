struct RigidBody{T} <: AbstractDynamicalSystem{T}
    I₁::T
    I₂::T
    I₃::T
end

RigidBody{T}() where {T} = RigidBody{T}(1.6, 1.0, 2 / 3)  # Defaults

# Ground truth equations of motion
function (rb::RigidBody)(du, u, p, t)
    (; I₁, I₂, I₃) = rb
    y₁, y₂, y₃ = u
    du[1] = (1 / I₃ - 1 / I₂) * y₃ * y₂
    du[2] = (1 / I₁ - 1 / I₃) * y₁ * y₃
    du[3] = (1 / I₂ - 1 / I₁) * y₂ * y₁
    return nothing
end

function initial_conditions(::RigidBody{T}) where {T}
    ϕ = 0.5 + rand()
    return T[cos(ϕ), 0.0, sin(ϕ)]
end

function get_trajectories(
    system::RigidBody{T},
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
    systemBF = RigidBody{BigFloat}()
    trajectories = []
    for _ = 1:N
        time_series = integrate_trajectory(
            systemBF;
            seconds,
            dt,
            transient_seconds,
            solver,
            reltol,
            abstol,
            u0 = initial_conditions(systemBF),
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

function constraints(u::AbstractVector{T}, t::T, ::RigidBody) where {T}
    y₁, y₂, y₃ = u
    return [T(0.5) * (y₁^2 + y₂^2 + y₃^2)]  # Casimir function
end

function constraints_jacobian(u::AbstractVector{T}, t::T, ::RigidBody) where {T}
    y₁, y₂, y₃ = u
    return [
        y₁ y₂ y₃
    ]
end

function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::RigidBody{T},
    ::Val{1},
) where {T}
    return get_mlp(3 => 3, hidden_layers, hidden_width, augment_dim, activation, T)
end

function rhs_neural(u, θ, t, re::Optimisers.Restructure, ::RigidBody, ::Val{1})
    return re(θ)(u)
end
