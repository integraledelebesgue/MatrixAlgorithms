module Factorization
export lup, display

using LinearAlgebra: SingularException, LowerTriangular, UpperTriangular, Diagonal
import Base.display

struct LUP{T}
    size::Int
    factorized::Matrix{T}
    L::Matrix{T}
    U::Matrix{T}
    p::Vector{Int}
end

function LUP(matrix::Matrix{T})::LUP where T <: Number
    shape = size(matrix)
    type = eltype(matrix)

    LUP{type}(
        shape[1],
        copy(matrix),
        zeros(T, shape...),
        zeros(T, shape...),
        collect(1:shape[1])
    )
end

function display(factorization::LUP{T})::Nothing where {T}
    println("LUP{Matrix{$T}} factorization:")
    println("L factor:")
    Base.display(factorization.L)

    println("U factor:")
    Base.display(factorization.U)

    println("Row permutation:")
    Base.display(factorization.p)
end

function swap!(vector::Vector{<:Number}, i::Int, j::Int)
    vector[i], vector[j] = vector[j], vector[i]
end

function swap_rows!(matrix::Matrix{<:Number}, i::Int, j::Int)
    matrix[i, :], matrix[j, :] = matrix[j, :], matrix[i, :]
end

function subtract_schur_complement!(matrix::Matrix{<:Number}, col::Int)
    a = matrix[col, col]
    v = @view(matrix[col+1:end, col])
    w = @view(matrix[col, col+1:end])
    submatrix = @view(matrix[col+1:end, col+1:end])

    v ./= a

    submatrix .-= v * w' 
end

function get_pivot(matrix::Matrix{T}, col::Int)::Tuple{Int, T} where {T}
    relative_row = @view(matrix[col:end, col]) .|> abs |> argmax
    row = relative_row + col - 1
    pivot = abs(matrix[row, col])
    
    (row, pivot)
end

function fill_triangles!(result::LUP)
    result.L .+= LowerTriangular(result.factorized)
    result.L .-= Diagonal(result.L)
    result.L .+= Diagonal(ones(size(result.L, 1)))
    result.U .+= UpperTriangular(result.factorized)
end

function lup(matrix::Matrix{<:Number})::LUP
    result = LUP(matrix)
    n = result.size

    for col ∈ 1:n-1
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