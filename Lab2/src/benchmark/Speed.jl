module Speed
export benchmark

using Base.Threads
using Pipe: @pipe
using Base.Iterators: map as lazy_map

function test(f::Function, data::Matrix{Float64})
    print(size(data, 1))
    @spawn @elapsed f(data)
end

function await_all(results)
    @views map!(fetch, results[3, :], results[3, :])
    results
end

function data(size::Int)::Matrix{Float64}
    rand(size, size)
end

function hstack(vectors)
    reduce(hcat, vectors)
end

function benchmark(functions::Vector{Function}, sizes::AbstractArray{Int, 1}, n_evals::Int)::Matrix{Float64}
    cases = lazy_map(
        n -> rand(n, n),
        sizes
    )

    results = @sync [
        [size(data, 1), i_fun, test(functions[i_fun], data)]
        for data in cases
        for _ in n_evals
        for i_fun in eachindex(functions)
    ]

    results |> hstack |> await_all |> transpose
end

end# module