module Imports
export @load_src_directories

using Base.Iterators: flatmap
using Pipe: @pipe

function subdirectories(path::String)::Vector{String}
    dirs = @pipe path |> 
        readdir(_, join=true) |> 
        filter(isdir)

    subdirs = @pipe dirs |> 
        flatmap(subdirectories, _) |> 
        collect

    return vcat(
        dirs, 
        subdirs
    )
end

function block(exprs::Vector{Expr})::Expr
    Expr(:block, exprs...)
end

function quote_push(path::String)::Expr
    :(push!(LOAD_PATH, $path))
end

macro load_src_directories(path::String)
    path |> 
    subdirectories .|> 
    quote_push |> 
    block |> 
    esc
end

end# module