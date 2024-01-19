module Inversion
export inv, inv!

using LoopVectorization: @turbo
using Base: @pure

MatrixOrView = Union{Matrix{Float64}, SubArray}


@pure function is_power_of_two(n::Int)::Bool
    @fastmath ==((log(2, n) .|> [floor, ceil])...)
end

function inv(matrix::Matrix{Float64})::Matrix{Float64}
    @assert is_power_of_two(size(matrix, 1))

    size(matrix, 1) == 1 && 
        return Base.inv(matrix[1])

    matrix |> 
        copy |> 
        inv!
end

function trivial_inv!(matrix::MatrixOrView)::MatrixOrView
    @inbounds a, c, b, d = matrix
    @fastmath inv_det = Base.inv(a * d - b * c)

    @inbounds @fastmath begin
        matrix[1, 1] = d * inv_det
        matrix[1, 2] = -b * inv_det
        matrix[2, 1] = -c * inv_det
        matrix[2, 2] = a * inv_det
    end

    return matrix
end

function inv!(matrix::MatrixOrView)::MatrixOrView
    n = size(matrix, 1)
    n_half = n ÷ 2

    if n <= 2
        return trivial_inv!(matrix)
    end

    upper = 1:n_half
    lower = n_half+1:n

    @inbounds @views begin
        m11 = matrix[upper, upper]
        m12 = matrix[upper, lower]
        m21 = matrix[lower, upper]
        m22 = matrix[lower, lower]
    end

    m11_inv = inv!(m11)

    @turbo @fastmath m22 .-= m21 * m11_inv * m12
    m22_inv = inv!(m22)

    let m11_inv_copy = copy(m11_inv)
        @turbo @fastmath m11_inv .= m11_inv_copy + m11_inv_copy * m12 * m22_inv * m21 * m11_inv_copy
        @turbo @fastmath m12 .= -m11_inv_copy * m12 * m22_inv
        @turbo @fastmath m21 .= -m22_inv * m21 * m11_inv_copy
    end

    return matrix
end

end# module
