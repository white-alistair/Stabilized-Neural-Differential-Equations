module StabilizedNDEs

using Optimisers,
    OrdinaryDiffEq,
    SciMLSensitivity,
    Flux,
    Zygote,
    Optimisers,
    Parameters,
    ParameterSchedulers,
    DynamicalSystemsBase,
    ChaosTools,
    Printf,
    ArgParse,
    LinearAlgebra,
    StatsBase,
    Statistics,
    FastClosures,
    Random,
    DelimitedFiles,
    TOML,
    JLD2,
    Printf
import CairoMakie  # Keep namespace clean

include("abstract_types.jl")
include("command_line.jl")
include("stabilized_ndes.jl")
include("time_series.jl")
include("data.jl")
include("multiple_shooting.jl")
include("evaluate.jl")
include("losses.jl")
include("scheduler.jl")
include("optimiser.jl")
include("adjoints.jl")
include("neural_nets.jl")
include("learning_curves.jl")
include("train.jl")
include("serialization.jl")
include("wrap_angle.jl")
include("io.jl")
include("relative_error.jl")

include("experiments/two_body_problem.jl")
include("experiments/rigid_body.jl")
include("experiments/converter.jl")
include("experiments/robot_arm.jl")
include("experiments/double_pendulum.jl")

end
