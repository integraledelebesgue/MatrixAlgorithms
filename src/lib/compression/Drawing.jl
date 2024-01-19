module Drawing
export draw, save

import HierarchicalMatrices: HMatrix, Node, Trivial, Zero, Divided, Compressed

using Plots: heatmap, savefig
using Base: OneTo

function save(path::String)::Function
    img -> savefig(img, path)
end

function draw(matrix::HMatrix)::BitMatrix
    n = length(matrix.root.rows)
    m = length(matrix.root.cols)

    canvas = ones(Bool, n, m) |> BitMatrix
    mark!(canvas, matrix.root)
end

function mark!(matrix::BitMatrix, node::Node)::BitMatrix
    if node.state == Trivial || node.state == Compressed
        matrix[node.rows, node.cols] .= false
        
    elseif node.state == Divided
        mark!(matrix, node.children.ul)
        mark!(matrix, node.children.ur)
        mark!(matrix, node.children.ll)
        mark!(matrix, node.children.lr)
    end

    matrix
end

end# module