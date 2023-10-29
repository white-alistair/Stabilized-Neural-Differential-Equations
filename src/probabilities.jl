function get_hellinger_distance(
    system,
    tr1::AbstractMatrix{T},
    tr2::AbstractMatrix{T},
    N::Int,
) where {T}
    ρ1, ρ2 = get_invariant_measures(system, tr1, tr2, N)
    return hellinger(ρ1, ρ2)
end

function get_invariant_measures(
    ::AbstractDynamicalSystem,
    tr1::AbstractMatrix{T},
    tr2::AbstractMatrix{T},
    N::Int,
) where {T}
    return get_invariant_measures(tr1, tr2, N)
end

function get_invariant_measures(
    ::DoublePendulum,
    tr1::AbstractMatrix{T},
    tr2::AbstractMatrix{T},
    N::Int,
) where {T}
    @. tr1[1:2, :] = wrap_angle(tr1[1:2, :])
    @. tr2[1:2, :] = wrap_angle(tr2[1:2, :])
    return get_invariant_measures(tr1, tr2, N)
end

function get_invariant_measures(
    tr1::AbstractMatrix{T},
    tr2::AbstractMatrix{T},
    N::Int,
) where {T}
    binning = get_fixed_rectangular_binning(tr1, tr2, N)

    iv1 = invariantmeasure(Dataset(tr1'), binning)
    iv2 = invariantmeasure(Dataset(tr2'), binning)

    # Unpack the probabilities and bins
    p1, o1 = iv1.ρ, iv1.to.bins
    p2, o2 = iv2.ρ, iv2.to.bins

    d1 = Dict(o1, p1)
    d2 = Dict(o2, p2)
    all_outcomes = o1 ∪ o2

    ρ1 = [get(d1, outcome, zero(T)) for outcome in all_outcomes]
    ρ2 = [get(d2, outcome, zero(T)) for outcome in all_outcomes]

    return ρ1, ρ2
end

function get_fixed_rectangular_binning(
    tr1::AbstractMatrix{T},
    tr2::AbstractMatrix{T},
    N,
) where {T}
    # Surely there's a cleaner way of doing this
    ϵmin = Tuple([
        min(min1, min2) for (min1, min2) in
        zip(mapslices(minimum, tr1, dims = 2), mapslices(minimum, tr2, dims = 2))
    ])
    ϵmax = Tuple([
        max(max1, max2) for (max1, max2) in
        zip(mapslices(maximum, tr1, dims = 2), mapslices(maximum, tr2, dims = 2))
    ])
    return FixedRectangularBinning(ϵmin, ϵmax, N)
end

function Base.Dict(outcomes::AbstractVector, probabilities::Probabilities)
    return Dict(zip(outcomes, probabilities))
end
