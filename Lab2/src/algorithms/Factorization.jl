module Factorization
export lup, display

using LinearAlgebra: SingularException, LowerTriangular, UpperTriangular, Diagonal
using LoopVectorization: @turbo
using Base: @assume_effects
using Base.Threads: @spawn
import Base.display

abstract type LUP end

struct DecomposedLUP <: LUP
    size::Int
    factorized::Matrix{Float64}
    L::Matrix{Float64}
    U::Matrix{Float64}
    p::Vector{Int}
end

function DecomposedLUP(matrix::Matrix{Float64})::DecomposedLUP
    shape = size(matrix)

    DecomposedLUP(
        shape[1],
        copy(matrix),
        zeros(shape...),
        zeros(shape...),
        Vector(1:shape[1])
    )
end

struct InplaceLUP <: LUP
    size::Int
    factorized::Matrix{Float64}
    p::Vector{Int}
end

function InplaceLUP(matrix::Matrix{Float64})::InplaceLUP
    n = size(matrix, 1)

    InplaceLUP(
        n,
        copyto!(similar(matrix), matrix),
        Vector(1:n)
    )
end

function display(factorization::DecomposedLUP)::Nothing
    println("LUP factorization:")
    println("L factor:")
    Base.display(factorization.L)

    println("U factor:")
    Base.display(factorization.U)

    println("Row permutation P:")
    Base.display(factorization.p)
end

function swap!(vector::Vector{Int}, i::Int, j::Int)
    @inbounds begin
        tmp = vector[i]
        vector[i] = vector[j]
        vector[j] = tmp
    end
end

function swap_rows!(matrix::Matrix{Float64}, i::Int, j::Int)
    n = size(matrix, 1)

    @inbounds @simd for col in 1:n
        tmp = matrix[i, col]
        matrix[i, col] = matrix[j, col]
        matrix[j, col] = tmp
    end
end

function scale_column!(matrix::Matrix{Float64}, col::Int)
    inv_element = inv(matrix[col, col])
    n = size(matrix, 1)

    @simd for i in col+1:n
        @inbounds @fastmath matrix[i, col] *= inv_element
    end
end

@polly function subtract_schur_complement!(matrix::Matrix{Float64}, col::Int)
    n = size(matrix, 1)
    
    for j in col+1:n
        @simd for i in col+1:n
            @inbounds @fastmath matrix[i, j] -= matrix[i, col] * matrix[col, j]
        end
    end
end

function get_pivot(matrix::Matrix{Float64}, col::Int)::Tuple{Int, Float64}
    n = size(matrix, 1)
    @inbounds pivot = abs(matrix[col, col])
    pivot_row = col

    @simd for row in col+1:n 
        @inbounds element_abs = abs(matrix[row, col])

        if element_abs > pivot
            pivot = element_abs
            pivot_row = row
        end
    end

    return (pivot_row, pivot)
end

function fill_upper_triangle!(source::Matrix{Float64}, destination::Matrix{Float64})
    @inbounds @views destination .= UpperTriangular(source)
end

function fill_lower_triangle!(source::Matrix{Float64}, destination::Matrix{Float64})
    @inbounds @views destination .= LowerTriangular(source)
end

function set_ones_diagonal!(matrix::Matrix{Float64})
    n = size(matrix, 1)
    step = n + 1
    stop = n ^ 2
    @turbo @views matrix[1:step:stop] .= 1.0
end

function fill_triangles!(result::DecomposedLUP)
    @sync begin
        @spawn fill_upper_triangle!(result.factorized, result.U)
        @spawn fill_lower_triangle!(result.factorized, result.L)
    end

    set_ones_diagonal!(result.L)
end

@assume_effects :total !:nothrow function lup(matrix::Matrix{Float64}; decompose::Bool = false)::LUP
    result = decompose ? 
        DecomposedLUP(matrix) :
        InplaceLUP(matrix)
    
    n = result.size

    for col ∈ 1:n-1
        pivot_row, pivot = get_pivot(result.factorized, col)

        pivot ≈ 0.0 && throw(SingularException(col))

        if pivot_row != col
            swap!(result.p, col, pivot_row)
            swap_rows!(result.factorized, col, pivot_row)
        end

        scale_column!(result.factorized, col)
        subtract_schur_complement!(result.factorized, col)
    end

    decompose && fill_triangles!(result)

    return result
end

end# module