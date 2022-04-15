push!(LOAD_PATH,"../src/")
using Bender
using Documenter
using PlutoStaticHTML

"""
Run all Pluto notebooks (".jl" files) in notebook directory and write output to Markdown files.
"""
function build_notebooks()
    println("Building tutorials")
    dir = joinpath(pkgdir(Bender), "docs", "src", "notebooks")
    # Evaluate notebooks in the same process to avoid having to recompile from scratch each time.
    # This is similar to how Documenter and Franklin evaluate code.
    # Note that things like method overrides may leak between notebooks!
    use_distributed = false
    output_format = documenter_output
    bopts = BuildOptions(dir; use_distributed, output_format)
    build_notebooks(bopts)
    return nothing
end

build_notebooks()

Home = "Home" => "index.md"
functionindex = "functionindex" => "functionindex.md"
FAexample = "Example: Feedback Alignment" => "notebooks/FA_notebook.md"

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


