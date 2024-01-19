module HierarchicalMatrices
export hmatrix, HMatrix

using LinearAlgebra: svd, Diagonal
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

macro sum(children, field)
    quote
        $children.ul.$field + 
        $children.ur.$field + 
        $children.ll.$field + 
        $children.lr.$field
    end |> esc
end

@enum State begin
    Divided
    Compressed
    Trivial
    Zero
end

struct SVD{T<:Number}
    U::Matrix{T}
    S::Vector{T}
    Vt::Matrix{T}
end

struct Node{T}
    state::State
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    svd::Option{Union{SVD{T}, Matrix{T}}}
    children::Option{Children}
    error::T
end

function Node(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int}, rank::Int, tolerance::T)::Node{T} where {T}
    is_zero(matrix, rows, cols) && return Node{T}(
        Zero, 
        rows,
        cols,
        nothing, 
        nothing,
        zero(T)
    )
    
    is_trivial(rows, cols) && return Node{T}(
        Trivial,
        rows,
        cols,
        matrix[rows, cols],
        nothing,
        zero(T)
    )

    decomposition = svd(@view matrix[rows, cols])

    if abs(decomposition.S[min(rank + 1, end)]) < tolerance
        truncated = SVD{T}(
            decomposition.U[:, 1:rank], 
            decomposition.S[1:rank], 
            decomposition.Vt[1:rank, :]
        )

        return Node{T}(
            Compressed,
            rows,
            cols,
            truncated,
            nothing,
            l2(recompose(truncated) - matrix[rows, cols])
        )
    end

    from_split(rows_cols::RangePair)::Node{T} = Node(matrix, rows_cols..., rank, tolerance)
    
    subranges = split(rows, cols)
    children = @flatmap(from_split, subranges, Node{T})
    total_error = @sum(children, error)

    Node{T}(
        Divided,
        rows,
        cols,
        nothing,
        children,
        total_error
    )
end

function is_trivial(rows::UnitRange{Int}, cols::UnitRange{Int})::Bool
    length(rows) <= 2 || length(cols) <= 2
end

@polly function is_zero(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int})::Bool where {T}
    for j in cols, i in rows
        @inbounds !iszero(matrix[i, j]) && return false
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

    HMatrix(root, rank, tolerance, root.error)
end

function hmatrix(matrix::Matrix{T}, rank::Int, tolerance::T) where {T}
    HMatrix(matrix, rank, tolerance)
end

function recompose(dec::SVD{T})::Matrix{T} where {T}
    return dec.U * Diagonal(dec.S) * dec.Vt
end

function l2(matrix::Matrix{T})::T where {T}
    foldl(
        (acc, element) -> acc + element ^ 2,
        matrix,
        init=0.0
    )
end

end# module
