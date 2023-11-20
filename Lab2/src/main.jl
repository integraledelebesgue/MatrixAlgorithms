using LinearAlgebra: lu
using LinearAlgebra: det as ldet
using LinearAlgebra: inv as a_inv
using Base.Threads
using DataFrames: DataFrame
using CSV

push!(LOAD_PATH, @__DIR__)
using Imports
Imports.@load_src_directories(".")

using Factorization: lup, display
using Determinant: det
using Inversion: inv as m_inv

using Speed: benchmark, Result

function to_dataframe(result::Result)::DataFrame
    DataFrame(result.data, result.header)
end 

const path = "data/flops.csv"

function save(df::DataFrame)
    open(path, read=true, truncate=true) do io
        CSV.write(io, df)
    end
end

function force_precompilation()::Nothing
    a = rand(8, 8)

    @sync begin
        @spawn lup(a, decompose=true)
        @spawn det(a)
        @spawn m_inv(a)
    end

    nothing
end

function main()
    force_precompilation()

    functions = [lup, ldet, m_inv]
    domain = 2 .^ collect(2:11)

    benchmark(functions, domain, 1, :flops) |> 
        to_dataframe |> 
        save
end

main() |> display

