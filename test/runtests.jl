using Pkg: Pkg
using NormalizingFlowFilters
using Test
using TestReports
using Aqua
using Documenter

include("test_assimilate_data.jl")

ts = @testset ReportingTestSet "" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(NormalizingFlowFilters; ambiguities=false)
        Aqua.test_ambiguities(NormalizingFlowFilters)
    end

    include("test_assimilate_data.jl")

    # Set metadata for doctests.
    DocMeta.setdocmeta!(
        NormalizingFlowFilters,
        :DocTestSetup,
        :(using NormalizingFlowFilters, Test);
        recursive=true,
    )

    # Run doctests.
    doctest(NormalizingFlowFilters; manual=true)

    # Run examples.
    examples_dir = joinpath(@__DIR__, "..", "examples")
    for example in readdir(examples_dir)
        example_path = joinpath(examples_dir, example)
        @show example_path
        orig_project = Base.active_project()
        @testset "Example: $(example)" begin
            if isdir(example_path)
                Pkg.activate(example_path)
                Pkg.develop(; path=joinpath(@__DIR__, ".."))
                Pkg.instantiate()
            end
            script_path = joinpath(example_path, "main.jl")
            try
                include(script_path)
                println("Included script_path")
            finally
                Pkg.activate(orig_project)
            end
        end
    end
end

outputfilename = joinpath(@__DIR__, "..", "report.xml")
open(outputfilename, "w") do fh
    print(fh, report(ts))
end
exit(any_problems(ts))
