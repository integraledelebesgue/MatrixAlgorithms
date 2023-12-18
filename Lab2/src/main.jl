using Base: compact
using LinearAlgebra: lu
using LinearAlgebra: det as ldet
using LinearAlgebra: inv as a_inv
using Base.Threads
using DataFrames: DataFrame
using CSV
using BenchmarkTools: @benchmark

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

const path = "data/huge_flops.csv"

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

function compare(my::Function, lib::Function, title::String)::Nothing
    a = rand(1024, 1024)
    
    my_time = @elapsed my(a)
    lib_time = @elapsed lib(a)
    
    println("Comparing $title for 1024x1024:")
    println("  My: $my_time s")
    println("  Library: $lib_time s")
    println()
end

function main()
    force_precompilation()

    functions = [lup, det, m_inv]
    domain = 2 .^ collect(2:14)

    benchmark(functions, domain, 1, :flops) |> 
        to_dataframe |> 
        save

    # compare(lup, lu, "LU Factorization")
    # compare(det, ldet, "Determinant")
    # compare(m_inv, inv, "Inversion")
end

main() |> display

