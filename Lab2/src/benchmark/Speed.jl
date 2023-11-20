module Speed
export benchmark, Result

using Base.Threads
using GFlops

struct Result
    header::Vector{Symbol}
    data::Matrix{Float64}
end

function hstack(vectors::Channel{Vector{Float64}})::Matrix{Float64}
    reduce(hcat, vectors)
end

function cases(sizes::AbstractArray{Int, 1})::Channel{Matrix{Float64}}
    Channel{Matrix{Float64}}() do channel
        for s in sizes
            put!(channel, rand(s, s))
        end
    end
end

const headers::Dict{Symbol, Vector{Symbol}} = Dict(
    :time => [:size, :function, :time],
    :flops => [:size, :function, :add, :mul]
)

function benchmark(functions::Vector{Function}, sizes::AbstractArray{Int, 1}, n_evals::Int, variant::Symbol)::Result
    data = Channel{Vector{Float64}}() do results
        for data in cases(sizes)
            for (i, f) in enumerate(functions)
                @threads :dynamic for _ in 1:n_evals
                    if variant === :time
                        time = @elapsed f(data)
                        put!(results, [size(data, 1), i, time])
                    elseif variant === :flops
                        counter = @count_ops f(data)
                        put!(results, [size(data, 1), i, counter.add64, counter.mul64])
                    end
                end
            end
        end
    end

    Result(
        headers[variant],
        data |> hstack |> transpose
    )
end

end# module