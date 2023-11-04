module Factorization
export lup, display

using LinearAlgebra: SingularException, LowerTriangular, UpperTriangular, Diagonal
using LoopVectorization: @turbo
using Base: @assume_effects
import Base.display

struct LUP
    size::Int
    factorized::Matrix{Float64}
    L::Matrix{Float64}
    U::Matrix{Float64}
    p::Vector{Int}
end

function LUP(matrix::Matrix{Float64})::LUP
    shape = size(matrix)

    LUP(
        shape[1],
        copy(matrix),
        zeros(shape...),
        zeros(shape...),
        collect(1:shape[1])
    )
end

function display(factorization::LUP)::Nothing
    println("LUP factorization:")
    println("L factor:")
    Base.display(factorization.L)

    println("U factor:")
    Base.display(factorization.U)

    println("Row permutation P:")
    Base.display(factorization.p)
end

function swap!(vector::Vector{Int}, i::Int, j::Int)
    @fastmath vector[i], vector[j] = vector[j], vector[i]
end

function swap_rows!(matrix::Matrix{Float64}, i::Int, j::Int)
    @turbo matrix[i, :], matrix[j, :] = matrix[j, :], matrix[i, :]
end

function subtract_schur_complement!(matrix::Matrix{Float64}, col::Int)
    a = matrix[col, col]
    v = @view(matrix[col+1:end, col])
    w = @view(matrix[col, col+1:end])
    submatrix = @view(matrix[col+1:end, col+1:end])

    @turbo v ./= a

    @turbo submatrix .-= v * w' 
end

function get_pivot(matrix::Matrix{Float64}, col::Int)::Tuple{Int, Float64}
    relative_row = @view(matrix[col:end, col]) .|> abs |> argmax
    row = relative_row + col - 1
    pivot = abs(matrix[row, col])
    
    (row, pivot)
end

function fill_triangles!(result::LUP)
    @inbounds begin
    result.L .+= LowerTriangular(result.factorized)
    result.L .-= Diagonal(result.L)
    result.L .+= Diagonal(ones(size(result.L, 1)))
    result.U .+= UpperTriangular(result.factorized)
    end
end

@assume_effects :total function lup(matrix::Matrix{Float64})::LUP
    result = LUP(matrix)
    n = result.size

    @inbounds for col ∈ 1:n-1
        pivot_row, pivot = get_pivot(result.factorized, col)

        if pivot ≈ 0.0
            throw(SingularException(col))
        end

        swap!(result.p, col, pivot_row)
        swap_rows!(result.factorized, col, pivot_row)

        subtract_schur_complement!(result.factorized, col)
    end

    fill_triangles!(result)

    return result
end

end# module