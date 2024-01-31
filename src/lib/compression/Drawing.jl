module Drawing
export draw, save, sparsity_pattern

import HierarchicalMatrices: HMatrix, Node, Trivial, Zero, Divided, Compressed

using Plots: heatmap, plot, scatter!, savefig, Plot
using Base: OneTo

using Images

function save(path::String)::Function
    img -> save(img, path)
end

function save(img::Plot, path::String)
    savefig(img, path)
end

const Image::Type = Array{Gray{Float64}, 2}

function save(img::Image, path::String)
    Images.save(path, img)
end

function draw(matrix::HMatrix)::Image
    n = length(matrix.root.rows)
    m = length(matrix.root.cols)

    canvas = trues(n, m)

    mark!(canvas, matrix.root)
    # reverse!(canvas, dims=1)

    # heatmap(canvas, legend=:none, axis=:off, cmap=:grays, aspectratio=:equal, title=title)
    Gray.(canvas)
end

function fill_svd!(matrix::BitMatrix, node::Node)
    rank = size(node.svd.S, 1)
    rows = node.rows
    cols = node.cols

    row_start = first(rows)
    row_end = row_start + rank - 1
    row_last = last(rows)

    matrix[row_start:row_end, cols] .= false
    matrix[row_last, cols] .= false

    col_start = first(cols)
    col_end = col_start + rank - 1
    col_last = last(cols)

    matrix[rows, col_start:col_end] .= false
    matrix[rows, col_last] .= false
end

function mark!(matrix::BitMatrix, node::Node)::BitMatrix
    if node.state == Trivial
        matrix[node.rows, node.cols] .= false

    elseif node.state == Compressed
        fill_svd!(
            matrix, 
            node
        )

    elseif node.state == Divided
        mark!(matrix, node.children.ul)
        mark!(matrix, node.children.ur)
        mark!(matrix, node.children.ll)
        mark!(matrix, node.children.lr)
    end

    matrix
end

function sparsity_pattern(mat::Matrix{<:Number})::Plot
    n = size(mat, 1)

    mask = mat .!= zero(eltype(mat))
    nonzero = sum(mask)

    plt = plot(
        title="$nonzero non-zero",
        size=(500, 500)
    )

    for index in findall(mask)
        scatter!(
            plt, 
            [index[2]], 
            n .- [index[1]], 
            legend=false, 
            aspect_ratio=:equal,
            markersize=3,
            color=:lightblue
        )
    end

    return plt
end

end# module