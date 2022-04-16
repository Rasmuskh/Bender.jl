push!(LOAD_PATH,"../src/")
using Bender
using Documenter

Home = "Home" => "index.md"
functionindex = "functionindex" => "functionindex.md"
FAexample = "Example: Feedback Alignment" => "notebooks/FAexample.md"

PAGES = [
    Home,
    functionindex,
    FAexample
    ]

makedocs(
        sitename = "Bender.jl",
        modules  = [Bender],
        format=Documenter.HTML(;
            canonical="https://rasmuskh.github.io/Bender.jl/",
            # Using MathJax3 since Pluto uses that engine too.
            mathengine=Documenter.MathJax3(),
            prettyurls=get(ENV, "CI", "false") == "true",
        ),
        pages=PAGES
        )

         
	deploydocs(;
	    repo="github.com/Rasmuskh/Bender.jl",
)


