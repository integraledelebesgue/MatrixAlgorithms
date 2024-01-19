module DeepMind
export multiply

using Base: split as base_split

function is_power_of(n::Int, base::Int)::Bool
    log_n = log(base, n)
    log_n ≈ floor(log_n)
end

function multiply(a::Matrix{<:Number}, b::Matrix{<:Number})::Matrix{<:Number}
    n, m = size(a)
    m_b, k = size(b)

    if m !== m_b
        throw(ArgumentError("Matrix sizes don't match"))
    end
    
    @assert is_power_of(n, 4) "n = $n"
    @assert is_power_of(m, 5) "m = $m"
    @assert is_power_of(k, 5) "k = $k"

    multiply_recursively(a, b)
end

function identifier(name::Symbol, i::Int, j::Int)::Symbol
    Symbol("$name$i$j")
end

macro divide(mtrx, n_row_blocks, n_col_blocks, row_block_width, col_block_width)
    declarations = [
        :($(identifier(mtrx, i, j)) = @view($mtrx[block_range($i, $j, $row_block_width, $col_block_width)...]))
        for i in 1:n_row_blocks
        for j in 1:n_col_blocks
    ]

    declarations |>
    block |> 
    esc
end

function split(delimiter::Char)::Function
    text -> base_split(text, delimiter)
end

function block(exprs::Vector{Expr})::Expr
    Expr(:block, exprs...)
end

macro declare_from_file(path::String)
    open(read, path, lock=true) |> 
    String |>
    split('\n') |>
    filter(!isempty) .|>
    Meta.parse |>
    block |>
    esc
end

function index_range(i::Int, width::Int)::UnitRange{Int}
    start = (i - 1) * width + 1
    stop = i * width
    start:stop
end

function block_range(i::Int, j::Int, row_width::Int, col_width::Int)::Tuple{UnitRange{Int}, UnitRange{Int}}
    index_range(i, row_width), index_range(j, col_width)
end

macro assign(mtrx, n_row_blocks, n_col_blocks, row_block_width, col_block_width)
    declarations = [
        :(
            $mtrx[block_range($i, $j, $row_block_width, $col_block_width)...] += 
            $(identifier(mtrx, i, j))[1:$row_block_width, 1:$col_block_width]
        )
        for i in 1:n_row_blocks
        for j in 1:n_col_blocks
    ]

    declarations |> 
    block |>  
    esc
end

function multiply_recursively(a, b)
    n, m = size(a)

    if n <= 4 || m <= 5
        return a * b
    end

    n_chunk = n ÷ 4
    m_chunk = m ÷ 5

    @divide(a, 4, 5, n_chunk, m_chunk)
    @divide(b, 5, 5, m_chunk, m_chunk)

    @declare_from_file "./meta/h.txt"
    @declare_from_file "./meta/c.txt"

    c = zeros(n, m)

    @assign(c, 4, 5, n_chunk, m_chunk)

    return c
end

end# module
