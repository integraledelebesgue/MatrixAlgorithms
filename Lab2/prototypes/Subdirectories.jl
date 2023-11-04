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

const path = "/home/integraledelebesgue/Studies/V/MatrixAlgorithms/Lab2"

path |> subdirectories |> display
