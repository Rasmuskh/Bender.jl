push!(LOAD_PATH,"../src/")
using RogueLearning
using Documenter

makedocs(
         sitename = "RogueLearning.jl",
         modules  = [RogueLearning],
         pages=[
                "Home" => "index.md"
               ])
	deploydocs(;
	    repo="github.com/Rasmuskh/RogueLearning.jl",
)


