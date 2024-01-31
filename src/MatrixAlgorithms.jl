push!(LOAD_PATH, @__DIR__)
using Using

using LinearAlgebra: svd

@use lib.compression.HierarchicalMatrices: hmatrix
@use lib.permutation.Permutations: Method, permute!, permutation, MinimalDegree, CuthillMcKee, ReversedCuthillMcKee
@use lib.generation.RandomMatrices: rand_sparse
@use lib.compression.Drawing: draw, sparsity_pattern, save
@use lib.generation.MethodMatrices: fem_3d

const directory = "images"

function save_sparsity_pattern(mat::Matrix{<:Number}, destination::String)
    mat |>
    sparsity_pattern |>
    save(destination)
end

function save_compression(mat::Matrix{<:Number}, destination::String)
    rank = 3
    threshold = 0.05

    hmat = hmatrix(mat, rank, threshold)

    draw(hmat) |> 
    save(destination)
end

function process(n::Int, method::Method)
    matrix = fem_3d(n)

    save_sparsity_pattern(matrix, "$(directory)/fem_$(n).png") |> display
    save_compression(matrix, "$(directory)/fem_$(n)_compressed.png") |> display

    permute!(
        matrix,
        permutation(matrix, method)
    )

    method_name = replace(
        lowercase("$(method)"),
        "(" => "",
        ")" => ""
    )

    save_sparsity_pattern(matrix, "$(directory)/fem_$(n)_$(method_name)_permuted.png") |> display
    save_compression(matrix, "$(directory)/fem_$(n)_$(method_name)_permuted_compressed.png") |> display
end

function generate()
    methods = [
        MinimalDegree(),
        CuthillMcKee(),
        ReversedCuthillMcKee()
    ]

    sizes = 2:4

    for method in methods, size in sizes
        process(size, method)
    end
end

function sample()
    m = rand_sparse(256, 0.1)
    rank = 3
    threshold = 0.1

    m |> 
        sparsity_pattern |> 
        save("images/sample.png")

    m_compressed = hmatrix(m, rank, threshold)

    draw(m_compressed) |>
        save("images/rand_sparse_compressed.png")

    p = permutation(m, MinimalDegree())

    permute!(m, p) |> 
        sparsity_pattern |> 
        save("images/rand_sparse_permuted.png")

    m_permuted_compressed = hmatrix(m, 3, 0.001)

    draw(m_permuted_compressed, title="Random (30% non-zero)") |>
        save("images/rand_sparse_permuted_hmatrix.png")
end

function main()
    # sample()
    # generate()
end

main() |> display
