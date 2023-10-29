"""
    StabilizedNDE

The field names correspond to the symbols in Equation 5 of the paper.

f should be a Callable with signature (u, θ, t), F and g should be Callables with 
signature (u, t), and γ should be a Real type.

The instantiated StabilizedNDE is then callable with signature (u, θ, t) and can be 
provided as the RHS to an ODEProblem.
"""
struct StabilizedNDE{A,B,C,D}
    f::A
    γ::B
    F::C
    g::D
end

function (snde::StabilizedNDE)(u, θ, t)
    (; f, γ, F, g) = snde
    return f(u, θ, t) .- γ * F(u, t) * g(u, t)
end

function StabilizedNDE(
    u0,
    t0,
    γ,
    restructure,
    augment_dim,
    system::AbstractDynamicalSystem{T},
    experiment_version,
) where {T}
    # Use the @closure macro from FastClosures.jl to avoid the Julia slow closure bug.
    f = @closure (u, θ, t) ->
        rhs_neural(u, θ, t, restructure, system, Val(experiment_version))
    F = @closure (u, t) -> begin
        J = constraints_jacobian(u, t, system)
        vcat(J' * inv(J * J'), zeros(T, augment_dim))
    end
    g = @closure (u, t) -> constraints(u, t, system) .- constraints(u0, t0, system)
    return StabilizedNDE(f, γ, F, g)
end
