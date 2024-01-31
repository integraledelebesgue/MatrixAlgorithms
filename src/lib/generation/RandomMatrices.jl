module RandomMatrices
export rand_sparse

using Random: shuffle!
using LinearAlgebra: det

function rand_sparse(size::Int, fullness::Float64)::Matrix{Float64}
    a = zeros(size, size)
    m = (Int ∘ floor)(size ^ 2 * fullness)

    @view(a[1:m]) .= rand(m)

    shuffle!(a)
end

function rand_nonsingular(size::Int)::Matrix{Float64}
    while true
        a = rand(size, size)
        det(a) ≉ 0.0 && return a
    end
end

function rand_3d_grid_fem(base_size::UInt)::Matrix{Float64}

end

end# module