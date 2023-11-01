push!(LOAD_PATH, @__DIR__)

using Test: @test
using Binet: multiply as b_multiply
using Strassen: multiply as s_multiply
using DeepMind: multiply as d_multiply
using Benchmark: benchmark, to_csv

using Base.Threads: @spawn

const df_destination = "data/times.csv"

function test(n::Int, m::Int, k::Int)
    a = rand(n, m)
    b = rand(m, k)
    
    solution = a * b
    
    binet = b_multiply(a, b)
    @test(solution ≈ binet) |> display

    strassen = s_multiply(a, b)
    @test(solution ≈ strassen) |> display
    
    deepmind = d_multiply(a, b)
    @test(solution ≈ deepmind) |> display
end

function main()
    # 2:3 |>
    # benchmark |>
    # to_csv(df_destination)

    # 1:3 .|>
    # (_ -> (rand(4, 5), rand(5, 5))) .|>
    # (arg -> @spawn(s_multiply(arg...))) .|>
    # fetch .|>
    # display

    s_multiply(rand(16, 16), rand(16, 16)) |> display
    @elapsed(s_multiply(rand(256, 256), rand(256, 256))) |> display 
end

main()

