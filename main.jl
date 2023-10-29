using InteractiveUtils
@info sprint(versioninfo)

using LibGit2
@info "HEAD = $(LibGit2.head("."))"

using StabilizedNDEs:
    parse_command_line,
    TwoBodyProblem,
    RigidBody,
    DCDCConverter,
    RobotArm,
    DoublePendulum,
    get_data,
    get_mlp,
    get_optimiser,
    get_scheduler,
    get_adjoint,
    train!,
    serialize,
    save_results,
    save_learning_curve
using Parameters, Random, LinearAlgebra

function main(args)
    #! format: off
    # Unpack command line args into variables in current scope
    @unpack job_id, rng_seed, NF = args
    # Experiment args
    @unpack experiment, experiment_version, stabilization_param = args
    # Data generation args
    @unpack dt, T, transient_seconds, data_solver, data_reltol, data_abstol = args
    # Data split args
    @unpack steps, n_train, n_valid, n_test = args
    # Solver args
    @unpack reltol, abstol, solver, sensealg, vjp, checkpointing = args
    # Neural net args
    @unpack hidden_layers, hidden_width, augment_dim, activation = args
    # Training args
    @unpack epochs, schedule_file, optimiser_rule, optimiser_hyperparams, patience, 
    time_limit, manual_gc = args
    # I/0
    @unpack verbose, show_plot, results_file, learning_curve_dir = args
    #! format: on

    Random.seed!(rng_seed)

    if experiment == :two_body_problem
        system = TwoBodyProblem{NF}()
    elseif experiment == :rigid_body
        system = RigidBody{NF}()
    elseif experiment == :converter
        system = DCDCConverter{NF}()
    elseif experiment == :robot_arm
        system = RobotArm{NF}()
    elseif experiment == :double_pendulum
        system = DoublePendulum{NF}()
    end

    # Set up the MLP
    θ, restructure = get_mlp(
        hidden_layers,
        hidden_width,
        augment_dim,
        activation,
        system,
        Val(experiment_version),
    )

    # Generate the training data
    # The SNDE object is bundled with the data, since the conserved quantity g depends on
    # the trajectory
    data = get_data(
        system,
        experiment_version,
        T,
        dt,
        transient_seconds,
        data_solver,
        data_reltol,
        data_abstol,
        n_train,
        n_valid,
        n_test,
        steps,
        stabilization_param,
        θ,
        restructure,
        augment_dim,
    )

    # Set up the optimiser and the schedule
    optimiser = get_optimiser(optimiser_rule, optimiser_hyperparams)
    scheduler = get_scheduler(schedule_file)

    # Set up the adjoint
    adjoint = get_adjoint(sensealg, vjp, checkpointing)

    # Train the model
    training_duration, learning_curve, min_val_epoch, min_val_loss, test_loss = train!(
        θ,
        data,
        epochs,
        optimiser,
        scheduler;
        solver,
        adjoint,
        reltol,
        abstol,
        patience,
        time_limit,
        verbose,
        show_plot,
        manual_gc,
    )

    # I/O
    save_results(
        results_file;
        job_id,
        experiment,
        experiment_version,
        hidden_layers,
        hidden_width,
        augment_dim,
        activation,
        stabilization_param,
        min_val_loss,
        test_loss,
        min_val_epoch,
        epochs,
        patience,
        training_duration,
        time_limit,
        dt,
        T,
        transient_seconds,
        steps,
        n_train,
        n_valid,
        n_test,
        optimiser_rule,
        optimiser_hyperparams = string(optimiser_hyperparams),
        schedule_file,
        reltol,
        abstol,
        rng_seed,
        NF,
        L2 = norm(θ, 2),
    )
    serialize(job_id, θ, restructure, stabilization_param)
    save_learning_curve(learning_curve, job_id, dir = learning_curve_dir)
end

args = parse_command_line(log = true)
main(args)
