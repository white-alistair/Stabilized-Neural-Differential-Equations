"""
    MSE(predicted_trajectory, target_trajectory)

Compute the mean-squared error between the predicted trajectory and the target trajectory.
"""
function MSE(predicted_trajectory, target_trajectory)
    return mean(abs2, predicted_trajectory[:, 2:end] .- target_trajectory[:, 2:end])  # Do not include u0
end
