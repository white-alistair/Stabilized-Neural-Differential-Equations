@testitem "Two Body Problem" begin
    using ForwardDiff, Random

    Random.seed!(1)

    T = Float64
    system = StabilizedNDEs.TwoBodyProblem{T}()
    u = rand(T, 4)
    t = rand(T)
    
    jac_forward_diff =
        ForwardDiff.jacobian(u -> StabilizedNDEs.constraints(u, t, system, Val(1)), u)
    jac_analytic = StabilizedNDEs.constraints_jacobian(u, t, system, Val(1))
    @test jac_analytic ≈ jac_forward_diff atol = 1e-6
end

@testitem "Rigid Body" begin
    using ForwardDiff, Random

    Random.seed!(1)

    T = Float64
    system = StabilizedNDEs.RigidBody{T}()
    u = rand(T, 3)
    t = rand(T)

    jac_forward_diff =
        ForwardDiff.jacobian(u -> StabilizedNDEs.constraints(u, t, system, Val(1)), u)
    jac_analytic = StabilizedNDEs.constraints_jacobian(u, t, system, Val(1))
    @test jac_analytic ≈ jac_forward_diff atol = 1e-6
end

@testitem "DC-to-DC Converter" begin
    using ForwardDiff, Random

    Random.seed!(1)

    T = Float64
    system = StabilizedNDEs.DCDCConverter{T}()
    u = rand(T, 3)
    t = rand(T)

    jac_forward_diff =
        ForwardDiff.jacobian(u -> StabilizedNDEs.constraints(u, t, system, Val(1)), u)
    jac_analytic = StabilizedNDEs.constraints_jacobian(u, t, system, Val(1))
    @test jac_analytic ≈ jac_forward_diff atol = 1e-6
end

@testitem "Robot Arm" begin
    using ForwardDiff, Random

    Random.seed!(1)

    T = Float64
    system = StabilizedNDEs.RobotArm{T}()
    u = rand(T, 3)
    t = rand(T)

    jac_forward_diff =
        ForwardDiff.jacobian(u -> StabilizedNDEs.constraints(u, t, system, Val(1)), u)
    jac_analytic = StabilizedNDEs.constraints_jacobian(u, t, system, Val(1))
    @test jac_analytic ≈ jac_forward_diff atol = 1e-6
end

@testitem "Double Pendulum" begin
    using ForwardDiff, Random

    Random.seed!(1)

    T = Float64
    system = StabilizedNDEs.DoublePendulum{T}(rand(T, 5)...)
    u = rand(T, 4)
    t = rand(T)

    jac_forward_diff =
        ForwardDiff.jacobian(u -> StabilizedNDEs.constraints(u, t, system, Val(1)), u)
    jac_analytic = StabilizedNDEs.constraints_jacobian(u, t, system, Val(1))
    @test jac_analytic ≈ jac_forward_diff atol = 1e-6
end
