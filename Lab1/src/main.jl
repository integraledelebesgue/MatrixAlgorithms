push!(LOAD_PATH, @__DIR__)

import Test: @test
import Binet: multiply as b_multiply
import Strassen: multiply as s_multiply
import DeepMind: multiply as d_multiply
import Benchmark: benchmark, to_csv

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
    2:5 |>
    benchmark |>
    to_csv(df_destination)
end

main()
