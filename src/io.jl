"""
    save_results(filename; kwargs...)

Write the results of the experiment to the given CSV file.

Each kwarg key-value pair corresponds to a column name and its value, respectively.
"""
function save_results(filename; kwargs...)
    # If the file doesn't exist already, create it and add the header row
    if !isfile(filename)
        col_names = [keys(kwargs)]
        open(filename, "w") do io
            writedlm(io, col_names, ',')
        end
    end

    # Format float values
    formatted_cols = []
    for col in values(kwargs)
        if col isa AbstractFloat
            push!(formatted_cols, @sprintf "%.2e" col)
        elseif col isa AbstractVector
            push!(formatted_cols, string(col))
        else
            push!(formatted_cols, col)
        end
    end

    # Write the data to the file
    open(filename, "a") do io
        writedlm(io, [formatted_cols], ',')
    end

    return nothing
end
