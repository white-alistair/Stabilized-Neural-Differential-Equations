"""
    get_optimiser(rule_type, hyperparameters)
    
Helper function for setting up an optimiser object.

See Optimisers.jl for full details of the optimisers and their hyperparameters.
"""
function get_optimiser(rule_type, hyperparameters)
    if rule_type == :Adam
        rule = Optimisers.Adam()
    elseif rule_type == :AdamW
        rule = Optimisers.AdamW()
    elseif rule_type == :Nesterov
        rule = Optimisers.Nesterov()
    end

    if !isempty(hyperparameters)
        rule = Optimisers.adjust(rule; hyperparameters...)
    end

    return rule
end
