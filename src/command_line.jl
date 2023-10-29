function parse_command_line(; log = false)
    settings = ArgParseSettings(autofix_names = true)

    @add_arg_table settings begin
        #! format: off
        "--experiment"
            help = "The name of the experiment to run"
            arg_type = Symbol
            required = true
        "--experiment-version", "--version"
            help = "If multiple versions of an experiment exist, specify which version to run"
            arg_type = Int
            default = 1
        "--NF", "--precision"   
            help = "The number format to use"
            arg_type = DataType
            default = Float64
        "--job-id"
            help = "Job ID used for serialization of experiment results"
            arg_type = String
            default = get(ENV, "SLURM_JOB_ID", "1")
        "--rng-seed", "--seed"
            help = "RNG seed"
            arg_type = Int
            default = 1
        "--stabilization-param", "--gamma"
            help = "The value of the stabilization parameter (0 for vanilla unstabilized NDE)"
            arg_type = Float64

        # Data generation args
        "--T"
            help = "Duration of each training trajectory"
            arg_type = BigFloat
            required = true
        "--dt"
            help = "Timestep for training trajectories"
            arg_type = BigFloat
            required = true
        "--transient-seconds", "--transient"
            help = "Transient integration period at the beginning of each trajectory"
            arg_type = BigFloat
            default = zero(BigFloat)
        "--data-solver"
            help = "Name of solver from OrdinaryDiffEq.jl for generating training data"
            arg_type = OrdinaryDiffEqAlgorithm
            default = Vern9()
        "--data-reltol"
            help = "Solver relative tolerance for generating training data"
            arg_type = BigFloat
            default = BigFloat(1e-24)
        "--data-abstol"
            help = "Solver absolute tolerance for generating training data"
            arg_type = BigFloat
            default = BigFloat(1e-24)

        # Data split args
        "--steps"
            help = "The number of timesteps per multiple shooting chunk"
            arg_type = Int
            required = true
        "--shuffle"
            help = "Whether to shuffle the chunks after doing multiple shooting"
            action = :store_true
        "--n-train"
            help = "Number of training trajectories"
            arg_type = Int
        "--n-valid"
            help = "Number of validation trajectories"
            arg_type = Int
            default = 0
        "--n-test"
            help = "Number of test trajectories"
            arg_type = Int
            default = 0

        # Neural net args
        "--hidden-layers", "--layers"
            help = "Number of hidden layers"
            arg_type = Int
            required = true
        "--hidden-width", "--width"
            help = "Width of hidden layers"
            arg_type = Int
            required = true
        "--augment-dim"
            help = "Number of extra dimensions for augmented neural ODEs"
            arg_type = Int
            default = 0
        "--activation"
            help = "Activation function"
            arg_type = Function
            default = relu

        # Training args
        "--epochs"
            help = "Total number of epochs"
            arg_type = Int
            required = true
        "--schedule-file", "--schedule"
            help = "Path to learning rate schedule config file"
            arg_type = String
            required = true
        "--optimiser-rule", "--opt"
            help = "Choice of optimiser from Optimisers.jl"
            arg_type = Symbol
            default = :Adam
        "--optimiser-hyperparams", "--opt-params"
            help = "Additional optimiser hyperparameters (comma-separated NAME=VALUE list)"
            arg_type = NamedTuple
            default = (;)
        "--patience"
            help = "Patience for early stopping"
            arg_type = Int
            default = typemax(Int)  # ~Inf
        "--time-limit", "--time"
            help = "Time limit for the training loop"
            arg_type = Float64
            default = Inf64
        "--manual-gc"
            help = "Whether to manually perform garbage collection after every epoch"
            action = :store_true

        # Solver args
        "--reltol"
            help = "Solver relative tolerance"
            arg_type = Float64
            default = 1e-6
        "--abstol"
            help = "Solver absolute tolerance"
            arg_type = Float64
            default = 1e-6
        "--solver"
            help = "Name of solver from OrdinaryDiffEq.jl"
            arg_type = OrdinaryDiffEqAlgorithm
            default = Tsit5()
        "--sensealg"
            help = "Name of sensitivity algorithm from SciMLSensitivity.jl"
            arg_type = Symbol
            default = :BacksolveAdjoint
        "--vjp"
            help = "Choice of AD for computing the vector-Jacobian product"
            arg_type = Symbol
            default = :ZygoteVJP
        "--checkpointing"
            action = :store_true

        # I/0
        "--verbose"
            help = "Whether to print loss for every training iteration"
            action = :store_true
        "--show-plot"
            help = "Whether to plot loss curve during training"
            action = :store_true
        "--results-file"
            help = "Where to store description of results"
            arg_type = String
            default = "results.csv"
        "--learning-curve-dir", "--lc-dir"
            help = "Directory in which to store learning curve"
            arg_type = String
            default = "learning_curves"
        #! format: on
    end

    args = parse_args(settings)
    if log
        log_args(args)
    end

    return args
end

function log_args(args)
    ordered_args = sort(collect(args); by = x -> x[1])
    for (arg_name, arg_value) in ordered_args
        @info "$arg_name = $arg_value"
    end
end

# Various functions for parsing custom types
# https://argparsejl.readthedocs.io/en/latest/argparse.html#parsing-to-custom-types
function eval_string(s)
    return eval(Meta.parse(s))
end

function ArgParse.parse_item(::Type{DataType}, type_name::AbstractString)
    return eval_string(type_name)
end

function ArgParse.parse_item(::Type{Function}, function_name::AbstractString)
    return eval_string(function_name)
end

function ArgParse.parse_item(::Type{OrdinaryDiffEqAlgorithm}, solver_name::AbstractString)
    return eval_string(solver_name * "()")
end

function ArgParse.parse_item(::Type{NamedTuple}, arg_string::AbstractString)
    items = eachsplit(arg_string, ",")
    names = [Symbol(strip.(split(item, "="))[1]) for item in items]
    args = [parse(Float32, (split(item, "=")[2])) for item in items]
    return (; zip(names, args)...)
end
