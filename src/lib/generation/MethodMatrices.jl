module MethodMatrices
export fem_3d

# function fem_3d(n_nodes::Int)::Matrix{Float64}
#     n = 2 ^ k
#     n_2 = n ^ 2
#     n_3 = n ^ 3

#     matrix = zeros(Int64, n_3, n_3)
    
#     for node in 1:n_3

#     end
# end

function fem_3d(k)
    n = 2^k
    n_2 = n^2 

    matrix = zeros(n^3, n^3)
    matrix_size = size(matrix)[1]

    for vertex in 1:matrix_size
        level = (vertex - 1) ÷ n_2
        rest = (vertex - 1) % n_2
        row = rest ÷ n
        col = rest % n

        matrix[vertex, vertex] = rand()
        if level > 0
            top_level_neighbor = vertex - n_2
            matrix[vertex, top_level_neighbor] = rand()
        end

        if level < n - 1
            bottom_level_neighbor = vertex + n_2
            matrix[vertex, bottom_level_neighbor] = rand()
        end

        if row > 0
            top_neighbor = vertex - n
            matrix[vertex, top_neighbor] = rand()
        end

        if row < n - 1
            bottom_neighbor = vertex + n
            matrix[vertex, bottom_neighbor] = rand()
        end

        if col > 0
            left_neighbor = vertex - 1
            matrix[vertex, left_neighbor] = rand()
        end

        if col < n - 1
            right_neighbor = vertex + 1
            matrix[vertex, right_neighbor] = rand()
        end
    end

    return matrix
end

end# module