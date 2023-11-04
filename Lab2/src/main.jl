push!(LOAD_PATH, @__DIR__)

using Imports
Imports.@load_src_directories(".")

using Factorization: lup, display

using LinearAlgebra: lu

function main()
    a = rand(10, 10)

    fac = lup(a)
    display(a[fac.p, :] .≈ fac.L * fac.U)
end

main()
