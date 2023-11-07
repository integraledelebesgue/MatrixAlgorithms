using LinearAlgebra: Diagonal, LowerTriangular, UpperTriangular
using LoopVectorization
using BenchmarkTools
using Pipe: @pipe
using Random: MersenneTwister


function set_diagonal_lib!(matrix::Matrix{Float64})::Nothing
    @inbounds begin
        matrix -= Diagonal(matrix)
        matrix += Diagonal(ones(size(matrix, 1)))
    end

    nothing
end

function set_diagonal_loop_simd!(matrix::Matrix{Float64})::Nothing
    n = size(matrix, 1)

    @simd for i in 1:n
        @inbounds matrix[i, i] = 1.0
    end

    nothing
end

function set_diagonal_loop_turbo!(matrix::Matrix{Float64})::Nothing
    n = size(matrix, 1)

    @turbo for i in 1:n
        matrix[i, i] = 1.0
    end

    nothing
end

function set_diagonal_broadcast_turbo!(matrix::Matrix{Float64})::Nothing
    n = size(matrix, 1)

    @turbo @views matrix[1:n+1:n^2] .= 1.0

    nothing
end

function set_triangle_lib!(source::Matrix{Float64}, destination::Matrix{Float64})::Nothing
    @inbounds destination .+= LowerTriangular(source)


    nothing
end

@polly function set_triangle_loop_polly!(source::Matrix{Float64}, destination::Matrix{Float64})::Nothing
    n = size(source, 1)

    for i in 1:n
        for j in 1:i
            @inbounds destination[i, j] = source[i, j]
        end
    end

    nothing
end

@polly function set_triangle_loop_polly_simd!(source::Matrix{Float64}, destination::Matrix{Float64})::Nothing
    n = size(source, 1)

    for i in 1:n
        @simd for j in 1:i
            @inbounds destination[i, j] = source[i, j]
        end
    end

    nothing
end

@polly function set_triangle_loop_polly_turbo!(source::Matrix{Float64}, destination::Matrix{Float64})::Nothing
    n = size(source, 1)

    for i in 1:n
        @turbo for j in 1:i
            destination[i, j] = source[i, j]
        end
    end

    nothing
end

function set_triangle_broadcast_turbo!(source::Matrix{Float64}, destination::Matrix{Float64})::Nothing
    n = size(source, 1)

    @inbounds @simd for i in 1:n
        @turbo @views destination[i, 1:i] .= source[i, 1:i]
    end

    nothing
end

function main(n::Int)
    diagonal_tests = [
        @benchmarkable set_diagonal_lib!(rand(MersenneTwister(0), $n, $n))
        @benchmarkable set_diagonal_loop_simd!(rand(MersenneTwister(0), $n, $n))
        @benchmarkable set_diagonal_loop_turbo!(rand(MersenneTwister(0), $n, $n))
        @benchmarkable set_diagonal_broadcast_turbo!(rand(MersenneTwister(0), $n, $n))
    ]

    triangle_tests = [
        @benchmarkable set_triangle_lib!($(rand(MersenneTwister(0), n, n)), $(zeros(n, n)))
        @benchmarkable set_triangle_loop_polly!($(rand(MersenneTwister(0), n, n)), $(zeros(n, n)))
        @benchmarkable set_triangle_loop_polly_simd!($(rand(MersenneTwister(0), n, n)), $(zeros(n, n)))
        @benchmarkable set_triangle_loop_polly_turbo!($(rand(MersenneTwister(0), n, n)), $(zeros(n, n)))
        @benchmarkable set_triangle_broadcast_turbo!($(rand(MersenneTwister(0), n, n)), $(zeros(n, n)))
    ]

    println("Diagonal")
    diagonal_tests .|> run .|> display
    # println("Triangle:")
    # triangle_tests .|> run .|> display
end


@pipe ARGS |> first |> parse(Int, _) |> main