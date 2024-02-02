using Base: summarysize
push!(LOAD_PATH, @__DIR__)
using Using

using LinearAlgebra: svd
using DataFrames: DataFrame

@use lib.compression.HierarchicalMatrices: hmatrix, add, multiply, dense
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

const signature = [:matrix_size, :base, :minimal_degree, :cuthill_mckee, :reversed_cuthill_mckee]

function compare_sizes()
    local rank = 3
    local threshold = 0.1

    methods = [
        MinimalDegree(),
        CuthillMcKee(),
        ReversedCuthillMcKee()
    ]

    sizes = 2:4

    compare(size::Int)::Vector{Int} = begin
        GC.gc()
        matrix = fem_3d(size)

        byte_sizes = [
            summarysize(hmatrix(matrix, rank, threshold)); 
            [
                summarysize(hmatrix(permute!(
                    matrix,
                    permutation(matrix, method)
                ), rank, threshold))
                for method in methods
            ] 
        ]

        byte_sizes
    end

    data = pushfirst!.(
        map(compare, sizes), 
        2 .^ (3 .* sizes)
    )

    DataFrame(
        reduce(hcat, data)', 
        signature
    )
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

const rank = 2
const threshold = 0.01

function fail_info(result::Matrix{Float64}, valid::Matrix{Float64})::String
    diff = result - valid
    error_min, error_max = extrema(diff)
    error_total = sum(diff)

    "error ∈ < $error_min, $error_max >, total = $error_total"
end

function test_add(size::Int)
    print("test_add($size): ")

    a = rand(size, size)
    b = rand(size, size)
    
    valid = a + b

    result = add(
        hmatrix(a, rank, threshold),
        hmatrix(b, rank, threshold)
    ) |> dense
    
    @assert all(valid .≈ result) fail_info(result, valid)
    println("passed!")
end

function test_multiply(size::Int)
    print("test_multiply($size): ")

    a = rand(size, size)
    b = rand(size, size)
    
    valid = a * b

    result = multiply(
        hmatrix(a, rank, threshold),
        hmatrix(b, rank, threshold)
    ) |> dense
    
    @assert all(valid .≈ result) fail_info(result, valid)
    println("passed!")
end

function draw_sum(size::Int, density::Float64)
    local rank = 3
    local threshold = 0.5

    a = hmatrix(rand_sparse(size, density), rank, threshold)
    b = hmatrix(rand_sparse(size, density), rank, threshold)

    destination = directory * "/hmatrix_arithmetics/results"

    a |> 
        draw |> 
        save(destination * "/term1.png")

    b |> 
        draw |> 
        save(destination * "/term2.png")

    add(a, b) |> 
        draw |> 
        save(destination * "/sum.png")
end

function draw_product(size::Int, density::Float64)
    local rank = 3
    local threshold = 0.5

    a = hmatrix(rand_sparse(size, density), rank, threshold)
    b = hmatrix(rand_sparse(size, density), rank, threshold)

    destination = directory * "/hmatrix_arithmetics/results"

    a |> 
        draw |> 
        save(destination * "/factor1.png")

    b |> 
        draw |> 
        save(destination * "/factor2.png")

    multiply(a, b) |> 
        draw |> 
        save(destination * "/product.png")
end

function main()
    # sample()
    # generate()

    test_add(8)
    test_multiply(8)

    # test_add(512)
    # test_multiply(512)

    # compare_sizes()

    draw_sum(256, 0.01)
    draw_product(256, 0.01)
end

main() |> display
