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
    @test(solution == binet) |> display

    strassen = s_multiply(a, b)
    @test(solution == strassen) |> display
    
    deepmind = d_multiply(a, b)
    @test(solution == deepmind) |> display
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
    # test(16, 25, 25)
    # flops_benchmark(69, 21, 37) |> display

    # a = rand(16, 25)
    # b = rand(25, 25)

    # c_d = d_multiply(a, b)
    # c = a * b

    # display(c_d .== c)

    #d_multiply(rand(16, 25), rand(25, 25)) |> display
end

main()
