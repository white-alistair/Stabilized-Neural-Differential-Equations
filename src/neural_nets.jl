"""
    function get_mlp(
        (input_size, output_size)::Pair,
        hidden_layers,
        hidden_width,
        augment_dim = 0,
        activation = relu,
        T = Float32,
    )

Helper function for setting up a multilayer perceptron.
"""
function get_mlp(
    (input_size, output_size)::Pair,
    hidden_layers,
    hidden_width,
    augment_dim = 0,
    activation = relu,
    T = Float32,
)
    mlp = Chain(
        Dense(input_size + augment_dim => hidden_width, activation),                       # First hidden layer
        [Dense(hidden_width => hidden_width, activation) for _ = 1:(hidden_layers-1)]...,  # Remaining hidden layers
        Dense(hidden_width => output_size + augment_dim),                                  # Output layer
    )
    mlp = Flux._paramtype(T, mlp)
    θ, re = Optimisers.destructure(mlp)
    return θ, re
end
