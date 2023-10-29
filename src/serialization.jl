"""
    serialize(id, weights, restructure, gamma; dir = "serialization")

Serialize model weights using JLD2.
"""
function serialize(id, weights, restructure, gamma; dir = "serialization")
    mkpath(dir)
    file = "$(id).jld2"
    path = joinpath(dir, file)
    jldsave(path; weights, restructure, gamma)
end

"""
    deserialize(id; dir = "serialization")

Deserialize model weights using JLD2.
"""
function deserialize(id; dir = "serialization")
    mkpath(dir)
    file = "$(id).jld2"
    path = joinpath(dir, file)
    f = jldopen(path)
    return f["weights"], f["restructure"], f["gamma"]
end
