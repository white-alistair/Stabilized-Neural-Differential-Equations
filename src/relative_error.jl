"""
    get_relative_error(predicted::Matrix, ground_truth::Matrix)

Calculate the relative error ||̂u - u||₂ / ||u||₂.
"""
function get_relative_error(predicted::Matrix, ground_truth::Matrix)
    L2 = u -> norm(u, 2)
    num = mapslices(L2, predicted .- ground_truth, dims = 1)
    den = mapslices(L2, ground_truth, dims = 1)
    return vec(num ./ den)
end
