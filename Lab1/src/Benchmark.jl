module Benchmark
export benchmark, to_csv

using GFlops: @count_ops, Counter
using Binet: multiply as binet
using Strassen: multiply as strassen
using DeepMind: multiply as ai
using DataFrames: DataFrame
using CSV

const test_bases = (4, 5, 5)

const attributes = ["power", "algorithm", "time", "additions", "multiplications"]
const algorithms = Dict(
    binet => 0,
    strassen => 1,
    ai => 2
)

const adds = [:add16, :add32, :add64, :sub16, :sub32, :sub64]
const muls = [:mul16, :mul32, :mul64, :muladd16, :muladd32, :muladd64]

function additions(counter::Counter)::Int
    getproperty.([counter], adds) |> sum
end

function multiplications(counter::Counter)::Int
    getproperty.([counter], muls) |> sum
end

function atomic_test(algorithm::Function, power::Int, a::Matrix{Float64}, b::Matrix{Float64})::Vector{Float64}
    elapsed_time = @elapsed algorithm(a, b)
    # flops = @count_ops algorithm($a, $b)  # cursed function, doesn't work...
    
    [power, algorithms[algorithm], elapsed_time, 0 #=additions(flops)=#, 0 #=multiplications(flops)=#]
end

function test(power::Int)::Matrix{Float64}
    n, m, k = test_bases .^ power
    a = rand(n, m)
    b = rand(m, k)

    results = [atomic_test(alg, power, a, b) for alg in keys(algorithms)] 
    
    results |>
    hstack |>
    transpose
end

function benchmark(powers::UnitRange{Int})::DataFrame
    powers .|>
    test |>
    vstack |>
    dataframe
end

function dataframe(data::Matrix{Float64})::DataFrame
    DataFrame(data, attributes)
end

function hstack(collection::Vector{<:VecOrMat{Float64}})::Matrix{Float64}
    reduce(hcat, collection)
end

function vstack(collection::Vector{<:VecOrMat{Float64}})::Matrix{Float64}
    reduce(vcat, collection)
end

function to_csv(path::String)::Function
    df::DataFrame -> CSV.write(path, df)
end

end# module