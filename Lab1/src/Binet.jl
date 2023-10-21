module Binet
export multiply

import Base.Iterators: flatten

function pad(mtrx::Matrix{<:Number}, final_size::Int)::Matrix{<:Number}
    n, m = size(mtrx)
    padded = zeros(final_size, final_size)
    padded[1:n, 1:m] .= mtrx
    padded
end

function next_multiplicity(n::Int)::Function
    x::Int -> 2 * x - x % n
end

function pad_to_common_square(divisor::Int, matrices::Matrix{<:Number}...)
    new_size = matrices .|> 
        size |> 
        flatten |> 
        maximum |>
        next_multiplicity(divisor)

    pad.(matrices, new_size)
end

function multiply(a::Matrix{<:Number}, b::Matrix{<:Number}, n_blocks::Union{Int, Symbol} = :auto)::Matrix{<:Number}
    n, m = size(a)
    m_B, p = size(b)

    if m != m_B
        throw(ArgumentError("Matrix sizes don't match"))
    end

    if n_blocks === :auto
        n_blocks = max(n, m, p) ÷ 2
    end

    a, b = pad_to_common_square(n_blocks, a, b)

    m, _ = size(a)
    block_size = m ÷ n_blocks

    c = zeros(m, m)

    for i in 1:n_blocks
        for j in 1:n_blocks
            for k in 1:n_blocks
                multiply_blocks!(a, b, c, i, j, k, block_size)
            end
        end
    end

    return c[1:n, 1:p]
end

function index_range(index::Int, size::Int)::UnitRange{Int}
    start = (index - 1) * size + 1 
    stop = start + size - 1
    start:stop
end

function multiply_blocks!(
        a::Matrix{<:Number}, 
        b::Matrix{<:Number}, 
        c::Matrix{<:Number}, 
        block_i::Int, 
        block_j::Int, 
        block_k::Int, 
        block_size::Int
)::Nothing
    for i in index_range(block_i, block_size)
        for j in index_range(block_j, block_size)
            for k in index_range(block_k, block_size)
                c[i, j] += a[i, k] * b[k, j]
            end
        end
    end
end

end# module