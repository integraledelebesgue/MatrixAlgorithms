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

function name(method::Method)::String
    replace(
        lowercase("$(method)"),
        "(" => "",
        ")" => ""
    )
end

function process(matrix::Matrix{<:Number}, method::Method)
    n = size(matrix, 1)

    save_sparsity_pattern(matrix, "$(directory)/fem_$(n).png")
    save_compression(matrix, "$(directory)/fem_$(n)_compressed.png")

    permute!(
        matrix,
        permutation(matrix, method)
    )

    method_name = name(method)

    save_sparsity_pattern(matrix, "$(directory)/fem_$(n)_$(method_name)_permuted.png")
    save_compression(matrix, "$(directory)/fem_$(n)_$(method_name)_permuted_compressed.png")
end

function generate()
    methods = [
        MinimalDegree(),
        CuthillMcKee(),
        ReversedCuthillMcKee()
    ]

    sizes = 2:4

    println("Generation started")

    for size in sizes
        matrix = fem_3d(size)

        for method in methods
            process(matrix, method)
            println("  $(name(method)) ($size, $size) finished!")
        end

        GC.gc()
    end

    println("Finished!")
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

    draw(m_permuted_compressed) |>
        save("images/rand_sparse_permuted_hmatrix.png")
end

function main()
    # sample()
    generate()
end

main() |> display
