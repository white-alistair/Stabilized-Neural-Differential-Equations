"""
    multiple_shooting(prob, time_series::TimeSeries; steps::Int, shuffle = false)

Given a TimeSeries object, split it up into non-overlapping chunks of length steps.

Return a vector of 3-tuples, each of which contains an ODEProblem object, the times of the 
chunk, and the values of the chunk.
"""
function multiple_shooting(prob, time_series::TimeSeries; steps::Int, shuffle = false)
    (; times, trajectory) = time_series
    start_indexes = collect(1:steps:length(time_series)-steps)
    if shuffle
        shuffle!(start_indexes)
    end
    return @views [(prob, times[i:i+steps], trajectory[:, i:i+steps]) for i in start_indexes]
end
