struct TimeSeries{T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    times::V
    trajectory::M
    TimeSeries{T}(times::AbstractVector{T}, trajectory::AbstractMatrix{T}) where {T} =
        size(times)[end] != size(trajectory)[end] ?
        throw(DimensionMismatch("number of times and observations do not match")) :
        new{T,typeof(times), typeof(trajectory)}(times, trajectory)
end

function TimeSeries{T}(times::AbstractVector, trajectory::AbstractMatrix) where {T}
    TimeSeries{T}(T.(times), T.(trajectory))
end

Base.getindex(ode_solution::TimeSeries, i::Int) = ode_solution.trajectory[:, i]
Base.getindex(ode_solution::TimeSeries, I::UnitRange) = ode_solution.trajectory[:, I]
Base.firstindex(ode_solution::TimeSeries) = 1
Base.lastindex(ode_solution::TimeSeries) = size(ode_solution.trajectory)[2]
Base.length(ode_solution::TimeSeries) = length(ode_solution.times)
