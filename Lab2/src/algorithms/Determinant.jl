module Determinant

using LinearAlgebra: SingularException
using LoopVectorization: @turbo
using Base: @assume_effects

function swap_rows!(matrix::Matrix{Float64}, i::Int, j::Int)::Nothing
    @turbo @fastmath matrix[i, :], matrix[j, :] = matrix[j, :], matrix[i, :]
    nothing
end

function get_pivot(matrix::Matrix{Float64}, col::Int)::Tuple{Int, Float64}
    @inbounds begin
        relative_row = @view(matrix[col:end, col]) .|> abs |> argmax
        row = relative_row + col - 1
        pivot = matrix[row, col]
    end
    
    (row, pivot)
end

function divide_row!(matrix::Matrix{Float64}, row::Int)::Nothing
    @turbo @fastmath matrix[row, :] ./= matrix[row, row]
    nothing
end

function reduce_column!(matrix::Matrix{Float64}, col::Int)::Nothing
    n = size(matrix, 1)

    @simd for row in col+1:n
        @turbo matrix[row, :] .-= matrix[row, col] * matrix[col, :]
    end

    nothing
end

@assume_effects :total function det(matrix::Matrix{Float64})::Float64
    n = size(matrix, 1)
    matrix = copy(matrix)
    pivots = Vector{Float64}(undef, n)
    pivots[end] = 1.0

    for row_col in 1:n-1
        pivot_row, pivot = get_pivot(matrix, row_col)

        if pivot ≈ 0.0
            return 0.0
        end

        pivots[row_col] = pivot

        if pivot_row != row_col
            pivots[end] *= -1.0
        end

        swap_rows!(matrix, row_col, pivot_row)

        divide_row!(matrix, row_col)
        reduce_column!(matrix, row_col)
    end

    return @fastmath prod(pivots) * matrix[end, end]
end

end# module