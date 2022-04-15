push!(LOAD_PATH,"../src/")
using Bender
using Documenter

Home = "Home" => "index.md"
functionindex = "functionindex" => "functionindex.md"
FAexample = "Example: Feedback Alignment" => "FA_notebook.jl.html"

PAGES = [
    Home,
    functionindex,
    FAexample
    ]

makedocs(
         sitename = "Bender.jl",
         modules  = [Bender],
         pages=PAGES
         )
	deploydocs(;
	    repo="github.com/Rasmuskh/Bender.jl",
)


