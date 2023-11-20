module Determinant

using LoopVectorization: @turbo
using Base: @assume_effects
using LinearAlgebra: similar

function swap_rows!(matrix::Matrix{Float64}, i::Int, j::Int)::Nothing
    n = size(matrix, 1)

    @inbounds @simd for col in 1:n
        tmp = matrix[i, col]
        matrix[i, col] = matrix[j, col]
        matrix[j, col] = tmp
    end

    nothing
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

function divide_row!(matrix::Matrix{Float64}, row::Int)::Nothing
    @inbounds inv_pivot = inv(matrix[row, row])
    n = size(matrix, 1)

    @simd for i in row+1:n
        @inbounds @fastmath matrix[row, i] *= inv_pivot
    end

    nothing
end

@polly function reduce_column!(matrix::Matrix{Float64}, col::Int)::Nothing
    n = size(matrix, 1)

    @inbounds for i in col+1:n
        element = matrix[i, col]
        @simd for j in 1:n
            @fastmath matrix[i, j] -= element * matrix[col, i]
        end
    end

    nothing
end

@assume_effects :total function det(matrix::Matrix{Float64})::Float64
    matrix = copy(matrix)
    n = size(matrix, 1)
    n_rows_permuted = 0
    result = 1.0

    for row_col in 1:n
        pivot_row, pivot = get_pivot(matrix, row_col)

        pivot ≈ 0.0 && return 0.0

        @fastmath result *= pivot

        if pivot_row != row_col
            n_rows_permuted += 1
        end

        swap_rows!(matrix, row_col, pivot_row)

        divide_row!(matrix, row_col)
        reduce_column!(matrix, row_col)
    end

    sgn = isodd(n_rows_permuted) ? -1.0 : 1.0

    return @fastmath result * sgn
end

end# module
