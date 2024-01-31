module Permutations
export permute!, permutation, Method, MinimalDegree, CuthillMcKee, CuthillMcKee, ReversedCuthillMcKee

using Base: OneTo, visit, Perm
using DataStructures: PriorityQueue, enqueue!, dequeue!, Deque

using Base.Iterators: filter as lazy_filter

import Random: permute!
import Base: argmin

const Permutation = Vector{UInt}

abstract type Method end

struct MinimalDegree <: Method end
struct CuthillMcKee <: Method end
struct ReversedCuthillMcKee <: Method end


function permute!(mat::Matrix{T}, method::Method)::Matrix{T} where {T <: Number}
    permute!(
        mat, 
        permutation(mat, method)
    )
end

function permute!(mat::Matrix{T}, p::Permutation)::Matrix{T} where {T <: Number}
    permute!.(eachcol(mat), [p])
    permute!.(eachrow(mat), [p])
    
    mat
end


const Neighbourhood = Dict{UInt, Set{UInt}}

function neighbourhood(mat::Matrix{<:Number})::Neighbourhood
    n = size(mat, 1)
    
    indicator = mat .≉ zero(eltype(mat))

    nonzero = indicator |>
        eachrow .|>
        findall .|> 
        Set

    Dict(1:n .=> nonzero)
end

function min_degree(nb::Neighbourhood)::UInt
    argmin(
        length ∘ last, 
        nb
    ).first
end

function drop!(nb::Neighbourhood, i::UInt)
    pop!(nb, i)

    for set in values(nb)
        i in set && pop!(set, i)
    end
end

function permutation(mat::Matrix{<:Number}, ::MinimalDegree)::Permutation
    n = size(mat, 1)
    permutation = Vector{UInt}(undef, n)

    nb = neighbourhood(mat)

    for i in 1:n
        next = min_degree(nb)
        permutation[i] = next
        drop!(nb, next)
    end

    permutation
end


function indicator(mat::Matrix{<:Number})::BitMatrix
    ind = mat .!= zero(eltype(mat))
    
    for i in 1:size(mat, 1)
        ind[i, i] = false
    end

    ind
end

function adjacency(mat::Matrix{<:Number})::Vector{Vector{UInt}}
    mat |>
    indicator |>
    eachrow .|>
    findall
end

function permutation(mat::Matrix{<:Number}, ::CuthillMcKee)::Permutation
    n = size(mat, 1)

    visited = falses(n)
    adj = adjacency(mat)
    degrees = length.(adj)
    
    perm = Permutation()
    deque = Deque{UInt}()

    descending(vertices::Vector{UInt})::Vector{UInt} = 
        sort(vertices, by = i -> degrees[i])

    bfs!(start::UInt)::Nothing = begin
        push!(deque, start)

        while length(deque) > 0
            curr = popfirst!(deque)

            visited[curr] && continue
            visited[curr] = true

            push!(perm, curr)

            for neigh in descending(adj[curr])
                !visited[neigh] && push!(deque, neigh)
            end
        end

        nothing
    end

    for vertex in descending(UInt.(1:n))
        !visited[vertex] && bfs!(vertex)
    end

    perm
end

function permutation(mat::Matrix{<:Number}, ::ReversedCuthillMcKee)::Permutation
    permutation(mat, CuthillMcKee()) |> reverse
end

end# module
