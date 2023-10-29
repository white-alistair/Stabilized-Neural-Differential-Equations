struct RobotArm{T} <: AbstractDynamicalSystem{T} end

function endpoint(θ)
    θ₁, θ₂, θ₃ = θ
    return [cos(θ₁) + cos(θ₂) + cos(θ₃), sin(θ₁) + sin(θ₂) + sin(θ₃)]
end

function pose(t::T) where {T}
    return T[-sin(2π * t)/2π, 0.0]
end

function dpose(t::T) where {T}
    return T[-cos(2π * t), 0.0]
end

function (ra::RobotArm)(du, u, p, t)
    θ₁, θ₂, θ₃ = u
    G = [
        -sin(θ₁) -sin(θ₂) -sin(θ₃)
        cos(θ₁) cos(θ₂) cos(θ₃)
    ]
    du .= G' * inv(G * G') * dpose(t)
    return nothing
end

function initial_conditions(::RobotArm{T}) where {T}
    θ₀ = π / 4 + π / 8 * rand()  # [π/4, 3π/8]
    return T[θ₀, -θ₀, θ₀]
end

function get_trajectories(
    system::RobotArm{T},
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
    systemBF = RobotArm{BigFloat}()
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

function constraints(u, t, ::RobotArm)
    return endpoint(u) .- pose(t)
end

function constraints_jacobian(u::AbstractVector{T}, t::T, ::RobotArm{T}) where {T}
    θ₁, θ₂, θ₃ = u
    return [
        -sin(θ₁) -sin(θ₂) -sin(θ₃)
        cos(θ₁) cos(θ₂) cos(θ₃)
    ]
end

function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::RobotArm{T},
    ::Val{1},
) where {T}
    return get_mlp(8 => 3, hidden_layers, hidden_width, augment_dim, activation, T)
end

function rhs_neural(u, θ, t, re::Optimisers.Restructure, ::RobotArm, ::Val{1})
    θ₁, θ₂, θ₃ = u
    x = vcat(cos(θ₁), cos(θ₂), cos(θ₃), sin(θ₁), sin(θ₂), sin(θ₃), dpose(t))
    return re(θ)(x)
end
