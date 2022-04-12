push!(LOAD_PATH,"../src/")
using Bender
using Documenter

Home = "Home" => "index.md"
functionindex = "functionindex" => "functionindex.md"

PAGES = [
    Home,
    functionindex
    ]

makedocs(
         sitename = "Bender.jl",
         modules  = [Bender],
         pages=PAGES
         )
	deploydocs(;
	    repo="github.com/Rasmuskh/Bender.jl",
)


