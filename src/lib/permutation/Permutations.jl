module Permutations
export permute!, Method, MinimalDegree, CuthillMcKee, ReversedCuthillMcKee

using Base: OneTo, visit, Perm
using DataStructures: PriorityQueue, enqueue!, dequeue!

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
    n = size(mat, 2)

    for (old_row, new_row) in enumerate(p), i in 1:n
        @inbounds mat[old_row, i], mat[new_row, i] = mat[new_row, i], mat[old_row, i]
    end

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

function argmin(nb::Neighbourhood)::UInt
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
        next = argmin(nb)
        permutation[i] = next
        drop!(nb, next)
    end

    permutation
end


struct BFSControl
    queue::PriorityQueue{UInt, UInt}
    visited::BitVector
    neighbourhood::Neighbourhood

    BFSControl(mat::Matrix{<:Number}) =
        new(
            PriorityQueue{UInt, UInt}(),
            falses(size(mat, 1)),
            neighbourhood(mat)
        )
end

function not_finished(control::BFSControl)::Bool
    length(control.queue) > 0
end

function is_visited(vertex::UInt, control::BFSControl)::Bool
    control.visited[vertex]
end

function visit!(vertex::UInt, control::BFSControl)
    control.visited[vertex] = true
end

function neighbours(vertex::UInt, control::BFSControl)::Set{UInt}
    control.neighbourhood[vertex]
end

function descending_degrees(control::BFSControl)::Vector{UInt}
    sort(
        collect(keys(control.neighbourhood)), 
        by = x -> degree(x, control)
    )
end

function degree(vertex::UInt, control::BFSControl)::UInt
    length(control.neighbourhood[vertex])
end

function update!(vertex::UInt, control::BFSControl)
    deg = degree(vertex, control)

    if vertex ∉ keys(control.queue)
        enqueue!(control.queue, vertex, deg)
        return
    end

    if control.queue[vertex] > deg
        delete!(control.queue, vertex)
        enqueue!(control.queue, vertex, deg)
    end
end

function bfs!(control::BFSControl, perm::Permutation)
    while not_finished(control)
        curr = dequeue!(control.queue)
        
        is_visited(curr, control) && continue
        visit!(curr, control)

        push!(perm, curr)

        for nei ∈ neighbours(curr, control)
            is_visited(nei, control) && continue
            update!(nei, control)
        end
    end
end

function permutation(mat::Matrix{<:Number}, ::CuthillMcKee)::Permutation
    control = BFSControl(mat)
    perm = Permutation()

    for vertex in descending_degrees(control)
        is_visited(vertex, control) && continue

        update!(vertex, control)
        bfs!(control, perm)
    end

    perm
end

function permutation(mat::Matrix{<:Number}, ::ReversedCuthillMcKee)::Permutation
    permutation(mat, CuthillMcKee()) |> reverse
end

end# module
