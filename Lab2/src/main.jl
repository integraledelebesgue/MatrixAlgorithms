using Base: @assume_effects
using LinearAlgebra: lu
using LinearAlgebra: det as ldet
using LinearAlgebra: inv as a_inv
using BenchmarkTools
using Random: MersenneTwister
using Pipe: @pipe
using Base.Threads
using DataFrames: DataFrame
using CSV

push!(LOAD_PATH, @__DIR__)
using Imports
Imports.@load_src_directories(".")

using Factorization: lup, display
using Determinant: det
using Inversion: inv as m_inv

using Speed: benchmark

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

function to_dataframe(data)
    DataFrame(data, [:size, :function, :time])
end 

const path = "data/times.csv"

function save(df::DataFrame)
    open(path, read=true, truncate=true) do io
        CSV.write(io, df)
    end
end

function main()
    a = rand(8, 8)

    @sync begin
        @spawn lup(a, decompose=true)
        @spawn det(a)
        @spawn m_inv(a)
    end

    # display(all(a[fac.p, :] .≈ fac.L * fac.U))

    # size = 512

    # display(@benchmark lup($(rand(MersenneTwister(0), size, size))))
    # display(@benchmark lu($(rand(MersenneTwister(0), size, size))))

    # a = rand(Int8, 8, 8) .|> Float64
    # res = a * m_inv(a)

    # display(1 .- isapprox.(res, 0.0, atol=1e-10))

    # display(@benchmark size($a))

    # a |> det |> display
    # a |> ldet |> display

    # rand(10, 10) |> det |> display

    # display(@benchmark det($(rand(MersenneTwister(0), size, size))))
    # display(@benchmark ldet($(rand(MersenneTwister(0), size, size))))

    # display(@benchmark m_inv($(rand(MersenneTwister(0), size, size))))
    # display(@benchmark a_inv($(rand(MersenneTwister(0), size, size))))

    # functions = [lup, det, m_inv]

    # benchmark(
    #     functions,
    #     2 .^ collect(2:15),
    #     10
    # ) |> to_dataframe |> save

    @elapsed det(rand(32768, 32768))
end

main() |> display
