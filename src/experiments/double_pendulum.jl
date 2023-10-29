struct DoublePendulum{T} <: AbstractDynamicalSystem{T}
    m₁::T
    m₂::T
    l₁::T
    l₂::T
    g::T
end

DoublePendulum{T}() where {T} = DoublePendulum{T}(1.0, 1.0, 1.0, 1.0, 9.81)  # Defaults

# Ground truth equations of motion
@inline function dω₁(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂, g) = system
    θ₁, θ₂, ω₁, ω₂ = u
    return (
        -g * (2m₁ + m₂) * sin(θ₁) - m₂ * g * sin(θ₁ - 2θ₂) -
        2sin(θ₁ - θ₂) * m₂ * (ω₂^2 * l₂ + ω₁^2 * l₁ * cos(θ₁ - θ₂))
    ) / (l₁ * (2m₁ + m₂ - m₂ * cos(2(θ₁ - θ₂))))
end

@inline function dω₂(system::DoublePendulum, u::AbstractVector{T}) where {T}
    (; m₁, m₂, l₁, l₂, g) = system
    θ₁, θ₂, ω₁, ω₂ = u
    return (
        2sin(θ₁ - θ₂) *
        (ω₁^2 * l₁ * (m₁ + m₂) + g * (m₁ + m₂) * cos(θ₁) + ω₂^2 * l₂ * m₂ * cos(θ₁ - θ₂))
    ) / (l₂ * (2m₁ + m₂ - m₂ * cos(2(θ₁ - θ₂))))
end

function (system::DoublePendulum)(du, u, p, t)
    ω₁, ω₂ = u[3:4]
    du[1] = ω₁
    du[2] = ω₂
    du[3] = dω₁(system, u)
    du[4] = dω₂(system, u)
end

function initial_conditions(::DoublePendulum{T}) where {T}
    ϕ = π / 4 + π / 2 * rand()
    return T[ϕ, ϕ, 0.0, 0.0]
end

function get_trajectories(
    system::DoublePendulum{T},
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
) where {T}
    systemBF = DoublePendulum{BigFloat}()
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

function constraints(u, t, system::DoublePendulum{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, ω₁, ω₂ = u
    g = T(9.81)
    return [
        (
            T(0.5) * m₁ * l₁^2 * ω₁^2 +
            T(0.5) * m₂ * (l₁^2 * ω₁^2 + l₂^2 * ω₂^2 + 2l₁ * l₂ * ω₁ * ω₂ * cos(θ₁ - θ₂))  # Kinetic energy
        ) + (-(m₁ + m₂) * g * l₁ * cos(θ₁) - m₂ * g * l₂ * cos(θ₂)),                       # Potential energy
    ]
end

function constraints_jacobian(u, t, system::DoublePendulum{T}) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, ω₁, ω₂ = u
    g = T(9.81)
    # Have to construct the matrix like this due to Zygote bug
    # https://github.com/FluxML/Zygote.jl/issues/1413
    return [
        (-m₂*l₁*l₂*ω₁*ω₂*sin(θ₁ - θ₂)+(m₁+m₂)*g*l₁*sin(θ₁)) (m₂*l₁*l₂*ω₁*ω₂*sin(θ₁ - θ₂)+m₂*g*l₂*sin(θ₂)) (m₁*l₁^2*ω₁+m₂*l₁^2*ω₁+m₂*l₁*l₂*ω₂*cos(θ₁ - θ₂)) (m₂*l₂^2*ω₂+m₂*l₁*l₂*ω₁*cos(θ₁ - θ₂))
    ]
end

# EXPERIMENT 1: Second order neural ODE (purely data-driven)
function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::DoublePendulum{T},
    ::Val{1},
) where {T}
    return get_mlp(6 => 2, hidden_layers, hidden_width, augment_dim, activation, T)
end

function rhs_neural(
    u,
    θ,
    t,
    re::Optimisers.Restructure,
    ::DoublePendulum{T},
    ::Val{1},
) where {T}
    θ₁, θ₂, ω₁, ω₂ = u
    return vcat(ω₁, ω₂, re(θ)([ω₁, ω₂, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂)]))
end

# EXPERIMENT 2: UDE (Hybrid)
function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::DoublePendulum{T},
    ::Val{2},
) where {T}
    return get_mlp(6 => 1, hidden_layers, hidden_width, augment_dim, activation, T)
end

function rhs_neural(
    u,
    θ,
    t,
    re::Optimisers.Restructure,
    system::DoublePendulum{T},
    ::Val{2},
) where {T}
    (; m₁, m₂, l₁, l₂) = system
    θ₁, θ₂, ω₁, ω₂ = u

    mlp = x -> re(θ)(x)
    x = [ω₁, ω₂, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂)]
    g = T(9.81)

    dθ₁ = ω₁
    dθ₂ = ω₂
    dω₁ =
        (
            -g * (2m₁ + m₂) * sin(θ₁) - m₂ * g * sin(θ₁ - 2θ₂) -
            2sin(θ₁ - θ₂) * m₂ * (ω₂^2 * l₂ + ω₁^2 * l₁ * cos(θ₁ - θ₂))
        ) / (l₁ * (2m₁ + m₂ - m₂ * cos(2(θ₁ - θ₂))))
    dω₂ = mlp(x)[1]

    return [dθ₁, dθ₂, dω₁, dω₂]
end
