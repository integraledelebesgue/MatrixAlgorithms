module HierarchicalMatrices
using Base: eltypeof
export HMatrix

using LinearAlgebra: svd, SVD
import Base: view, map

const View{T} = SubArray{T, 2, Matrix{T}}
const MatrixOrView{T} = Union{Matrix{T}, View{T}}
const Option{T} = Union{Nothing, T}
const RangePair = NTuple{2, UnitRange{Int}}

struct Children{T}
    ul::Option{T}
    ur::Option{T}
    ll::Option{T}
    lr::Option{T}

    Children{T}(
        ul::Option{T} = nothing, 
        ur::Option{T} = nothing, 
        ll::Option{T} = nothing, 
        lr::Option{T} = nothing
    ) where {T} = new{T}(
        ul, ur, ll, lr
    )
end

macro flatmap(f, children, type)
    quote
        Children{$type}(
            $f($children.ul),
            $f($children.ur),
            $f($children.ll),
            $f($children.lr)
        )
    end |> esc
end

@enum State begin
    Divided
    Compressed
    Trivial
    Zero
end

struct Node{T}
    state::State
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    svd::Option{SVD{T}}
    children::Option{Children}
end

function Node(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int}, rank::Int, tolerance::T)::Node{T} where {T}
    is_trivial(rows, cols) && return Node{T}(
        Trivial,
        rows,
        cols,
        nothing,
        nothing
    )

    is_zero(matrix, rows, cols) && return Node{T}(
        Zero, 
        rows,
        cols,
        nothing, 
        nothing
    )

    decomposition = svd(@view matrix[rows, cols])
    decomposition.S[min(rank + 1, end)] < tolerance && return Node{T}(
        Compressed,
        rows,
        cols,
        decomposition,
        nothing
    )

    from_split(rows_cols::RangePair)::Node{T} = Node(matrix, rows_cols..., rank, tolerance)
    
    subranges = split(rows, cols)
    children = @flatmap(from_split, subranges, Node{T})

    Node{T}(
        Divided,
        rows,
        cols,
        nothing,
        children
    )
end

function is_trivial(rows::UnitRange{Int}, cols::UnitRange{Int})::Bool
    length(rows) <= 2 || length(cols) <= 2
end

function is_zero(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int})::Bool where {T}
    for j in cols, i in rows
        !iszero(matrix[i, j]) && return false
    end

    return true
end

function split(range::UnitRange{Int})::RangePair
    start, stop = extrema(range)
    centre = start + (stop - start) ÷ 2

    start:centre, centre+1:stop
end

function split(rows::UnitRange{Int}, cols::UnitRange{Int})::Children{RangePair}
    rows_first, rows_second = split(rows)
    cols_first, cols_second = split(cols)

    return Children{RangePair}(
        (rows_first, cols_first),
        (rows_first, cols_second),
        (rows_second, cols_first),
        (rows_second, cols_second)
    )
end

struct HMatrix{T<:Number}
    root::Node{T}
    rank::Int
    tolerance::T
    error::Float64
end

function HMatrix(matrix::Matrix{T}, rank::Int, tolerance::T) where {T}
    n, m = size(matrix)

    root = Node(
        matrix,
        1:n,
        1:m,
        rank, 
        tolerance
    )
    
    error = total_error(root)

    HMatrix(root, rank, tolerance, error)
end

function total_error(tree::Node)::Float64
    return 0.0
end

end# module