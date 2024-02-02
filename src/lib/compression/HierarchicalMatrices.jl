module HierarchicalMatrices
export hmatrix, HMatrix, dense, add, multiply

using LinearAlgebra: svd, Diagonal
using Base: OneTo as one_to
import Base: view, map
import Base: size

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

macro zip(children1, children2, type)
    quote
        Children{Tuple{$type, $type}}(
            ($children1.ul, $children2.ul),
            ($children1.ur, $children2.ur),
            ($children1.ll, $children2.ll),
            ($children1.lr, $children2.lr),
        )
    end |> esc
end

@enum State begin
    Divided
    Compressed
    Trivial
    Zero
end

struct SVD{T <: Number}
    U::Matrix{T}
    S::Vector{T}
    Vt::Matrix{T}
end

struct Node{T <: Number}
    state::State
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    svd::Option{Union{SVD{T}, Matrix{T}}}
    children::Option{Children}
    error::T
end

function Node(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
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

function node_unchecked(
        state::State, 
        rows::UnitRange{Int}, 
        cols::UnitRange{Int}, 
        data::Option{Union{SVD{T}, Matrix{T}}},
        children::Option{Children{Node{T}}},
        error::T
)::Node{T} where {T <: Number}
    Node{T}(state, rows, cols, data, children, error)
end

function node_from_slice(
        matrix::Matrix{T}, 
        rows::UnitRange{Int}, 
        cols::UnitRange{Int}, 
        original_rows::UnitRange{Int}, 
        original_cols::UnitRange{Int}, 
        rank::Int, 
        tolerance::T
)::Node{T} where {T <: Number}
    is_zero(matrix, rows, cols) && return Node{T}(
        Zero, 
        original_rows,
        original_cols,
        nothing, 
        nothing,
        zero(T)
    )

    is_trivial(rows, cols) && return Node{T}(
        Trivial,
        original_rows,
        original_cols,
        matrix[rows, cols],
        nothing,
        zero(T)
    )

    decomposition = svd(@view matrix[:, :])

    if abs(decomposition.S[min(rank + 1, end)]) < tolerance
        truncated = SVD{T}(
            decomposition.U[:, 1:rank], 
            decomposition.S[1:rank], 
            decomposition.Vt[1:rank, :]
        )

        return Node{T}(
            Compressed,
            original_rows,
            original_cols,
            truncated,
            nothing,
            l2(recompose(truncated) - matrix)
        )
    end

    from_split((rows_cols, original_rows_cols)::Tuple{RangePair, RangePair})::Node{T} = 
        node_from_slice(
            matrix, 
            rows_cols..., 
            original_rows_cols..., 
            rank, 
            tolerance
        )

    subranges = split(rows, cols)
    original_subranges = split(original_rows, original_cols)
    
    # beautiful overengineering *_* :
    children = @flatmap(  
        from_split, 
        @zip(subranges, original_subranges, Tuple{UnitRange{Int}, UnitRange{Int}}),
        Node{T}
    )

    # instead of this boilerplate:
    # rows_first, rows_second = split(rows)
    # cols_first, cols_second = split(cols)

    # original_rows_first, original_rows_second = split(original_rows)
    # original_cols_first, original_cols_second = split(original_cols)

    # children = Children{Node{T}}(
    #     node_from_slice(
    #         matrix,
    #         rows_first, cols_first,
    #         original_rows_first, original_cols_first,
    #         rank,
    #         tolerance
    #     ),
    #     node_from_slice(
    #         matrix,
    #         rows_first, cols_second,
    #         original_rows_first, original_cols_second,
    #         rank,
    #         tolerance
    #     ),
    #     node_from_slice(
    #         matrix,
    #         rows_second, cols_first,
    #         original_rows_second, original_cols_first,
    #         rank,
    #         tolerance
    #     ),
    #     node_from_slice(
    #         matrix,
    #         rows_second, cols_second,
    #         original_rows_second, original_cols_second,
    #         rank,
    #         tolerance
    #     )
    # )
    
    total_error = @sum(children, error)

    Node{T}(
        Divided,
        original_rows,
        original_cols,
        nothing,
        children,
        total_error
    )
end

function is_trivial(rows::UnitRange{Int}, cols::UnitRange{Int})::Bool
    length(rows) <= 2 || length(cols) <= 2
end

@polly function is_zero(matrix::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int})::Bool where {T <: Number}
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

function HMatrix(matrix::Matrix{T}, rank::Int, tolerance::T) where {T <: Number}
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

function hmatrix(matrix::Matrix{T}, rank::Int, tolerance::T) where {T <: Number}
    HMatrix(matrix, rank, tolerance)
end

function hmatrix_unchecked(root::Node{T}, rank::Int, tolerance::T)::HMatrix{T} where {T <: Number}
    HMatrix(root, rank, tolerance, root.error)
end

function size(mat::HMatrix{<:Number})::Tuple{Int, Int}
    root = mat.root
    (length(root.rows), length(root.cols))
end

function size(mat::HMatrix{<:Number}, dim::Int)::Int
    dim in (1, 2) ? 
        size(mat)[dim] : 
        throw(ArgumentError("Axis $dim is out of bounds for 2-dimensional HMatrix"))
end

function recompose(dec::SVD{T})::Matrix{T} where {T <: Number}
    return dec.U * Diagonal(dec.S) * dec.Vt
end

function l2(matrix::Matrix{T})::T where {T <: Number}
    foldl(
        (acc, element) -> acc + element ^ 2,
        matrix,
        init=0.0
    )
end

function dense(mat::HMatrix{T})::Matrix{T} where {T <: Number}
    dense(mat.root)
end

function dense(node::Node{T})::Matrix{T} where {T <: Number}
    if node.state == Trivial
        node.svd

    elseif node.state == Zero
        n = length(node.rows)
        zeros(T, n, n)

    elseif node.state == Compressed   
        recompose(node.svd)
    
    elseif node.state == Divided
        ul = dense(node.children.ul)
        ur = dense(node.children.ur)
        ll = dense(node.children.ll)
        lr = dense(node.children.lr)
        
        [ul ur; ll lr]
    end
end

function add_nonzero_leaves(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    if a.state == b.state == Trivial
        matrix = a.svd + b.svd
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == b.state == Compressed
        matrix = recompose(a.svd) + recompose(b.svd)
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == Trivial && b.state == Compressed
        matrix = a.svd + recompose(b.svd)
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == Compressed && b.state == Trivial
        matrix = recompose(a.svd) + b.svd
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )
    end
end

function add_nonzero_dense(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    matrix = dense(a) + dense(b)
    one_to_n = 1:size(matrix, 1)

    node_from_slice(
        matrix,
        one_to_n,
        one_to_n,
        a.rows,
        a.cols,
        rank,
        tolerance
    )
end

function add(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    if a.state == Zero
        deepcopy(b)

    elseif b.state == Zero
        deepcopy(a)

    elseif a.state == b.state == Divided
        children = Children{Node{T}}(
            add(a.children.ul, b.children.ul, rank, tolerance),
            add(a.children.ur, b.children.ur, rank, tolerance),
            add(a.children.ll, b.children.ll, rank, tolerance),
            add(a.children.lr, b.children.lr, rank, tolerance)
        )

        error = @sum(children, error)

        node_unchecked(
            Divided,
            a.rows,
            a.cols,
            nothing,
            children,
            error
        )

    elseif a.state != Divided && b.state != Divided
        add_nonzero_leaves(a, b, rank, tolerance)

    else
        add_nonzero_dense(a, b, rank, tolerance)
    end
end

function add(a::HMatrix{T}, b::HMatrix{T})::HMatrix{T} where {T <: Number}
    @assert size(a) == size(b) "Mismatched dimensions - $(size(a)) + $(size(b))"

    rank = min(a.rank, b.rank)
    tolerance = max(a.tolerance, b.tolerance)

    sum = add(a.root, b.root, rank, tolerance)
    hmatrix_unchecked(sum, rank, tolerance)
end

function multiply_nonzero_leaves(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    if a.state == b.state == Trivial
        matrix = a.svd * b.svd
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == b.state == Compressed
        matrix = recompose(a.svd) + recompose(b.svd)
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == Trivial && b.state == Compressed
        matrix = a.svd * recompose(b.svd)
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )

    elseif a.state == Compressed && b.state == Trivial
        matrix = recompose(a.svd) * b.svd
        one_to_n = 1:size(matrix, 1)

        node_from_slice(
            matrix,
            one_to_n,
            one_to_n,
            a.rows,
            a.cols,
            rank,
            tolerance
        )
    end
end

function multiply_nonzero_dense(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    matrix = dense(a) * dense(b)
    one_to_n = 1:size(matrix, 1)

    node_from_slice(
        matrix,
        one_to_n,
        one_to_n,
        a.rows,
        a.cols,
        rank,
        tolerance
    )
end

function multiply(a::Node{T}, b::Node{T}, rank::Int, tolerance::T)::Node{T} where {T <: Number}
    if a.state == Zero || b.state == Zero
        node_unchecked(
            Zero,
            a.rows,
            a.cols,
            nothing,
            nothing,
            zero(T)
        )

    elseif a.state == b.state == Divided
        # Recursive (Binet) block multiplication 
        # Order of arguments does matter.
        # rows and cols of a result Node are always inherited from the first argument

        children = Children{Node{T}}(
            add( #                  ul
                multiply(a.children.ul, b.children.ul, rank, tolerance),
                multiply(a.children.ur, b.children.ll, rank, tolerance),
                rank, 
                tolerance
            ),
            add( #                  ur
                multiply(a.children.ur, b.children.lr, rank, tolerance),
                multiply(a.children.ul, b.children.ur, rank, tolerance),
                rank, 
                tolerance
            ),
            add( #                  ll
                multiply(a.children.ll, b.children.ul, rank, tolerance),
                multiply(a.children.lr, b.children.ll, rank, tolerance),
                rank, 
                tolerance
            ),
            add( #                  lr
                multiply(a.children.lr, b.children.lr, rank, tolerance),
                multiply(a.children.ll, b.children.ur, rank, tolerance),
                rank, 
                tolerance
            )
        )

        error = @sum(children, error)

        node_unchecked(
            Divided,
            a.rows,
            a.cols,
            nothing,
            children,
            error
        )

    elseif a.state != Divided && b.state != Divided
        multiply_nonzero_leaves(a, b, rank, tolerance)

    else
        multiply_nonzero_dense(a, b, rank, tolerance)
    end
end

function multiply(a::HMatrix{T}, b::HMatrix{T})::HMatrix{T} where {T <: Number}
    @assert size(a, 2) == size(b, 1) "Mismatched dimensions - $(size(a)) * $(size(b))"

    rank = min(a.rank, b.rank)
    tolerance = max(a.tolerance, b.tolerance)

    product = multiply(a.root, b.root, rank, tolerance)
    hmatrix_unchecked(product, rank, tolerance)
end

end# module
