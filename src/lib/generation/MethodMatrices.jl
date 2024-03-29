module MethodMatrices
export fem_3d

function fem_3d(k::Int)::Matrix{Float64}
    n = 2 ^ k
    n_2 = n ^ 2 
    n_3 = n ^ 3

    matrix = zeros(Float64, n_3, n_3)

    assign!(row::Int, col::Int) = (matrix[row, col] = rand())

    for i in 1:n_3
        level = (i - 1) ÷ n_2
        remainder = (i - 1) % n_2
        row = remainder ÷ n
        col = remainder % n

        assign!(i, i)

        level > 0 && assign!(i, i - n_2)
        level < n - 1 && assign!(i, i + n_2)

        row > 0 && assign!(i, i - n)
        row < n - 1 && assign!(i, i + n)

        col > 0 && assign!(i, i - 1)
        col < n - 1 && assign!(i, i + 1)
    end

    matrix
end

end# module