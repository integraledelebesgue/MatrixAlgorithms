module Using
export @use

using Base.Iterators: peel, Rest

function split_dot(str::String)::Vector{String}
    split(str, ".")
end

const to_skip = ['(', ')']

function skip(str::String)::String
    filter(
        c -> c ∉ to_skip,
        str
    )
end

function match(str::String)
    splitted = str |> 
        skip |> 
        split_dot

    imports = pop!(splitted)
    location = joinpath(@__DIR__, splitted...)

    (location, imports)
end

function block(exprs::Vector{Expr})::Expr
    Expr(
        :block,
        exprs...
    )
end

macro use(path::Expr)
    location, imports = match("$path")

    push_call = :(push!(LOAD_PATH, $location))
    using_statement = Meta.parse("using $imports")

    [push_call, using_statement] |>
        block |>
        esc
end

end# module