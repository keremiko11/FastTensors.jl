using Test
using LinearAlgebra
using FastTensors

@testset "FastTensors.jl Test Suite" begin

    @testset "Tensor Creation & Norm" begin
        dims = (10, 8, 4)
        T = rand_tensor(Float64, dims)
        @test size(T) == dims
        @test ndims(T) == 3
        @test tensor_norm(T) > 0.0

        Z = zeros_tensor(Float32, (5, 5))
        @test size(Z) == (5, 5)
        @test tensor_norm(Z) == 0.0f0
    end

    @testset "Matrix-Vector Contraction Equivalence" begin
        # 2D x 1D contraction should match matrix-vector multiplication
        A = rand_tensor(Float64, (8, 6))
        v = rand_tensor(Float64, (6,))

        res = contract(A, v, 2, 1)
        expected = A.data * v.data

        @test size(res) == (8,)
        @test isapprox(res.data, expected, atol=1e-10)
    end

    @testset "HOSVD Decomposition" begin
        t = rand_tensor(Float64, (12, 10, 8))
        core, factors = hosvd(t, ranks=(6, 5, 4))

        @test size(core) == (6, 5, 4)
        @test length(factors) == 3
        @test size(factors[1]) == (12, 6)
        @test size(factors[2]) == (10, 5)
        @test size(factors[3]) == (8, 4)
    end

end
