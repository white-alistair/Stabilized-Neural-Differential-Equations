# https://ieeexplore.ieee.org/document/411098
struct DCDCConverter{T} <: AbstractDynamicalSystem{T}
    C₁::T
    C₂::T
    L₃::T
    period::T
    μ::T  # Duty ratio
end

DCDCConverter{T}() where {T} = DCDCConverter{T}(0.1, 0.2, 0.5, 3.0, 0.5)  # Defaults

function (system::DCDCConverter)(du, u, p, t)
    (; C₁, C₂, L₃, period, μ) = system
    V₁, V₂, I₃ = u
    s = (t % period) / period < μ ? 0 : 1

    du[1] = ((1 - s) * I₃) / C₁
    du[2] = (s * I₃) / C₂
    du[3] = (-(1 - s) * V₁ - s * V₂) / L₃

    return nothing
end

function initial_conditions(::DCDCConverter{T}) where {T}
    return rand(T, 3)
end

function get_trajectories(
    system::DCDCConverter{T},
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
    systemBF = DCDCConverter{BigFloat}()
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

function get_mlp(
    hidden_layers,
    hidden_width,
    augment_dim,
    activation,
    ::DCDCConverter{T},
    ::Val{1},
) where {T}
    return get_mlp(4 => 3, hidden_layers, hidden_width, augment_dim, activation, T)
end

function constraints(u, t, system::DCDCConverter{T}) where {T}
    (; C₁, C₂, L₃) = system
    V₁, V₂, I₃ = u
    return [T(0.5) * (C₁ * V₁^2 + C₂ * V₂^2 + L₃ * I₃^2)]
end

function constraints_jacobian(u, t, system::DCDCConverter)
    (; C₁, C₂, L₃) = system
    V₁, V₂, I₃ = u
    return [C₁ * V₁ C₂ * V₂ L₃ * I₃]
end

function rhs_neural(u, θ, t, re::Optimisers.Restructure, system::DCDCConverter, ::Val{1})
    (; period, μ) = system
    s = (t % period) / period < μ ? 0 : 1  # Switch ∈ {0,1}
    return re(θ)(vcat(u, s))
end
