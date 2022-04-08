push!(LOAD_PATH,"../src/")
using RogueLearning
using Documenter

Home = "Home" => "index.md"
functionindex = "functionindex" => "functionindex.md"

PAGES = [
    Home,
    functionindex
    ]

makedocs(
         sitename = "RogueLearning.jl",
         modules  = [RogueLearning],
         pages=PAGES
         )
	deploydocs(;
	    repo="github.com/Rasmuskh/RogueLearning.jl",
)


