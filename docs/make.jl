using Pkg: Pkg
using NormalizingFlowFilters
using Documenter

using Literate

const REPO_ROOT = joinpath(@__DIR__, "..")
const DOC_SRC = joinpath(@__DIR__, "src")
const DOC_STAGE = joinpath(@__DIR__, "stage")
const DOC_BUILD = joinpath(@__DIR__, "build")

function gen_runner_code(pth, in_dir, out_dir)
    return runner_code = """
           using Pkg: Pkg

           build_scripts = $build_scripts
           build_notebooks = $build_notebooks
           in_dir = $(repr(in_dir))
           out_dir = $(repr(out_dir))

           Pkg.activate(in_dir)
           Pkg.develop(; path=$(joinpath(@__DIR__, "..") |> repr))
           Pkg.add("Literate")
           Pkg.resolve()
           Pkg.instantiate()

           using Literate

           upd(content) = update_header(content, $(repr(pth)); build_notebooks, build_scripts)
           in_pth = joinpath(in_dir, "main.jl")

           # Copy other files over to out_dir.
           Base.Filesystem.cptree(in_dir, out_dir)
           rm(joinpath(out_dir, "main.jl"))

           include($(joinpath(@__DIR__, "utils.jl") |> repr))

           # Build outputs.
           Literate.markdown(in_pth, out_dir; name="index", preprocess=upd, execute=true)
           if build_notebooks
               Literate.notebook(in_pth, out_dir)
           end
           if build_scripts
               Literate.script(in_pth, out_dir)
           end
           """
end

# Move src files to staging area.
mkpath(DOC_STAGE)
for (root, dirs, files) in walkdir(DOC_SRC)
    println("Directories in $root: $dirs")
    rel_root = relpath(root, DOC_SRC)
    for dir in dirs
        stage = joinpath(DOC_STAGE, rel_root, dir)
        mkpath(stage)
    end
    println("Files in $root: $files")
    for file in files
        src = joinpath(DOC_SRC, rel_root, file)
        stage = joinpath(DOC_STAGE, rel_root, file)
        cp(src, stage)
    end
end

# Process examples and put them in staging area.
build_examples = true
build_notebooks = false
build_scripts = true
examples = ["Simple Usage n=2" => "simple-usage-2", "Simple Usage n=3" => "simple-usage-3"]
examples_markdown = []

mkpath(joinpath(DOC_STAGE, "examples"))
orig_project = Base.active_project()
for (ex, pth) in examples
    in_dir = joinpath(REPO_ROOT, "examples", pth)
    out_dir = joinpath(DOC_STAGE, "examples", pth)
    if build_examples
        push!(examples_markdown, ex => joinpath("examples", pth, "index.md"))

        runner_path = joinpath(mktempdir(), "runner.jl")
        runner_code = gen_runner_code(pth, in_dir, out_dir)

        open(runner_path, "w") do f
            write(f, runner_code)
        end
        cmd = `$(Base.julia_cmd()) -- "$(runner_path)"`

        @info "Testing  $(repr(ex)) at $(repr(pth)) with \"$(runner_path)\""

        proc = open(cmd, Base.stdout; write=true)
        wait(proc)
        if proc.exitcode != 0
            error("Failed to build example $(repr(ex)) at $(repr(pth))")
        end
    end
end

# Set metadata for doctests.
DocMeta.setdocmeta!(
    NormalizingFlowFilters,
    :DocTestSetup,
    :(using NormalizingFlowFilters, Test);
    recursive=true,
)
makedocs(;
    modules=[NormalizingFlowFilters],
    authors="Grant Bruer gbruer15@gmail.com and contributors",
    sitename="NormalizingFlowFilters.jl",
    source=DOC_STAGE,
    build=DOC_BUILD,
    format=Documenter.HTML(;
        repolink="https://github.com/tmp398243/tmp337502",
        canonical="https://tmp398243.github.io/tmp337502",
        edit_link="main",
        assets=String[],
        size_threshold=20 * 2^20,
    ),
    repo="github.com/tmp398243/tmp337502",
    pages=[
        "Home" => "index.md",
        "Examples" => examples_markdown,
        "Coverage" => "coverage/index.md",
    ],
    doctest=false,
)
