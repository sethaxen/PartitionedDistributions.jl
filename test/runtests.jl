using PartitionedDistributions
using Random
using Test
using Aqua
using JET

Random.seed!(14579)

@testset "PartitionedDistributions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(PartitionedDistributions)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(PartitionedDistributions; target_defined_modules = true)
    end

    include("testutils.jl")
    include("utils.jl")
    include("conditional_marginal.jl")
end
