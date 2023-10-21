push!(LOAD_PATH, @__DIR__)

import Test: @test
import GFlops: @count_ops
import Binet: multiply as b_multiply
import Strassen: multiply as s_multiply
import DeepMind: multiply as d_multiply

function test(n::Int, m::Int, k::Int)
    a = rand(1:100, n, m)
    b = rand(1:100, m, k)
    
    solution = a * b
    
    binet = b_multiply(a, b)
    @test solution == binet

    strassen = s_multiply(a, b)
    @test solution == strassen
    
    deepmind = d_multiply(a, b)
    @test solution == deepmind
end

function flops_benchmark(n::Int, m::Int, k::Int)
    a = rand(1:100, n, m)
    b = rand(1:100, m, k)

    binet = @count_ops b_multiply($a, $b)
    strassen = @count_ops s_multiply($a, $b)
    deepmind = @count_ops d_multiply($a, $b)

    binet, strassen, deepmind
end

function main()
    #test(420, 69, 69)
    flops_benchmark(30, 30, 30) |> display
end

main()
