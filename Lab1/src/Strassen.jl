module Strassen
export multiply

import Base.Iterators: flatten

const MatrixLike::Type = Union{Matrix, SubArray}

function pad(mtrx::Matrix{<:Number}, final_size::Int)::Matrix{<:Number}
    n, m = size(mtrx)
    padded = zeros(final_size, final_size)
    padded[1:n, 1:m] .= mtrx
    padded
end

function next_power_of_2(n::Int)::Int
    power = n |> log2 |> ceil |> Int
    2 ^ power
end

function pad_to_common_square(matrices::Matrix{<:Number}...)
    new_size = matrices .|> 
        size |> 
        flatten |> 
        maximum |>
        next_power_of_2

    pad.(matrices, new_size)
end

function multiply(a::Matrix{<:Number}, b::Matrix{<:Number}, threshold::Int = 2)::Matrix{<:Number}
    n, m = size(a)
    m_B, k = size(b)

    if m != m_B
        throw(ArgumentError("Matrix sizes don't match"))
    end

    if min(n, m, k) <= threshold
        return a * b
    end

    a, b = pad_to_common_square(a, b)

    m, _ = size(a)

    multiply_recursively(a, b)[1:n, 1:k]
end

function halves(mtrx::MatrixLike)::Tuple{UnitRange{Int}, UnitRange{Int}}
    n, _ = size(mtrx)
    n_half = n ÷ 2
    1:n_half, n_half+1:n
end

function identifier(name::Symbol, number::Int)::Symbol
    Symbol(String(name) * "$(number)")
end

macro divide(mtrx)
    code = quote
        $(identifier(mtrx, 11)) = @view($mtrx[first, first])
        $(identifier(mtrx, 12)) = @view($mtrx[first, second])
        $(identifier(mtrx, 21)) = @view($mtrx[second, first])
        $(identifier(mtrx, 22)) = @view($mtrx[second, second])
    end

    esc(code)
end 

function multiply_recursively(a::MatrixLike, b::MatrixLike, threshold::Int = 2)::Matrix{<:Number}
    n, _ = size(a)

    if n <= threshold
        return a * b
    end

    first, second = halves(a)

    @divide a
    @divide b

    p1 = multiply_recursively(a11 + a22, b11 + b22)
    p2 = multiply_recursively(a21 + a22, b11)
    p3 = multiply_recursively(a11, b12 - b22)
    p4 = multiply_recursively(a22, b21 - b11)
    p5 = multiply_recursively(a11 + a12, b22)
    p6 = multiply_recursively(a21 - a11, b11 + b12)
    p7 = multiply_recursively(a12 - a22, b21 + b22)

    c = zeros(n, n)

    c[first, first] += p1 + p4 - p5 + p7
    c[first, second] += p3 + p5
    c[second, first] += p2 + p4
    c[second, second] += p1 - p2 + p3 + p6

    return c
end

end# module