"""
    evaluate(θ, data, loss, solver, reltol, abstol)

Evaluate the loss for the parameters θ on the given data, i.e. validation or test data.
"""
function evaluate(θ, data, loss, solver, reltol, abstol)
    if isnothing(data)
        return NaN64
    end

    losses = Float64[]
    for (prob, times, target_trajectory) in data
        tspan = (times[1], times[end])
        u0 = target_trajectory[:, 1]
        prob = remake(prob; u0, tspan)
        sol = solve(prob, solver; p = θ, saveat = times, reltol, abstol)
        predicted_trajectory = Array(sol)
        push!(losses, loss(predicted_trajectory, target_trajectory))
    end
    
    return mean(losses)
end
