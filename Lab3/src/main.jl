push!(LOAD_PATH, @__DIR__)
using Using: @use

@use Decomposition.TSVD: TruncatedSVD, compress, display
@use Compression.HierarchicalMatrices: HMatrix

function l2(matrix)
    foldl(
        (acc, element) -> acc + element ^ 2,
        matrix
    )
end

function main()
    a = rand(10, 8)

    HMatrix(a, 5, 1e-5)
end

main() |> display

