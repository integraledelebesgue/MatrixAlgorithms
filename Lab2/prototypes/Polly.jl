using BenchmarkTools
using BenchmarkTools: @benchmark

using InteractiveUtils: @code_llvm
using Base.Threads

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 50

function do_something!(a::Matrix{Float64})
    m, n = size(a)

    for i in 1:m
        for j in 1:n
            @inbounds a[j, j] += i * j
        end
    end
end

@polly function do_something_polly!(a::Matrix{Float64})
    m, n = size(a)

    for i in 1:m
        for j in 1:n
            @inbounds a[j, j] += i * j
        end
    end
end

function multiply!(a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64})
    n = size(a, 1)

    for i in 1:n, j in 1:n, k in 1:n
        @inbounds c[i, j] += a[i, k] * b[k, j]
    end

    nothing
end

@polly function multiply_polly!(a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64})
    n = size(a, 1)

    for i in 1:n, j in 1:n, k in 1:n
        @inbounds c[i, j] += a[i, k] * b[k, j]
    end

    nothing
end

@polly function multiply_polly_simd!(a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64})\
    n = size(a, 1)

    for i in 1:n, j in 1:n
        @simd for k in 1:n
            @inbounds c[i, j] += a[i, k] * b[k, j]
        end
    end
end

const n = 500

x = rand(n, n)
y = rand(n, n)
z = zeros(size(x)...)

let normal, polly, polly_simd
    @sync begin 
        @spawn normal = @benchmark multiply!(x, y, z)
        @spawn polly = @benchmark multiply_polly!(x, y, z)
        @spawn polly_simd = @benchmark multiply_polly_simd!(x, y, z)
    end

    display(normal)
    display(polly)
    display(polly_simd)
end

# BenchmarkTools.Trial: 332 samples with 1 evaluation.
#  Range (min … max):  148.672 ms … 154.821 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     149.635 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   150.723 ms ±   1.753 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#         ▅██▂                                                     
#   ▃▁▂▃▅██████▆▃▄▃▃▁▄▄▂▃▂▃▂▄▃▁▃▂▃▃▃▃▃▃▄▄▂▁▅▃▄▂▄▄▄▃▂▃▃▃▁▃▂▄▃▅▆▄▅▄ ▃
#   149 ms           Histogram: frequency by time          154 ms <

#  Memory estimate: 0 bytes, allocs estimate: 0.
# BenchmarkTools.Trial: 331 samples with 1 evaluation.
#  Range (min … max):  148.779 ms … 197.283 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     149.730 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   151.276 ms ±   4.155 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#    █▁                                                            
#   ███▆▃▃▃▃▃▄▄▄▃▃▃▄▆▃▃▂▂▁▂▁▁▂▁▂▁▂▁▁▁▁▁▂▁▁▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▂ ▃
#   149 ms           Histogram: frequency by time          168 ms <

#  Memory estimate: 0 bytes, allocs estimate: 0.
# BenchmarkTools.Trial: 331 samples with 1 evaluation.
#  Range (min … max):  148.578 ms … 200.992 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     149.652 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   151.075 ms ±   3.897 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#    ▃█▁                                                           
#   ▄███▅▄▃▃▅▃▃▃▃▃▃▄▃▃▅▆▄▃▁▁▁▂▂▂▁▁▁▁▂▁▁▁▁▁▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▃
#   149 ms           Histogram: frequency by time          165 ms <

#  Memory estimate: 0 bytes, allocs estimate: 0.

# display(@code_llvm multiply!(x, y, z))
# println()
# display(@code_llvm multiply_polly!(x, y, z))
