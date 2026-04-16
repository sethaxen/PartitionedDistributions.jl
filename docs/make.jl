using PartitionedDistributions
using Distributions
using Documenter

DocMeta.setdocmeta!(PartitionedDistributions, :DocTestSetup, :(using PartitionedDistributions); recursive = true)

makedocs(;
    modules = [PartitionedDistributions],
    authors = "Seth Axen <seth@sethaxen.com>, Marco Bonici <bonici.marco@gmail.com>, and contributors",
    sitename = "PartitionedDistributions.jl",
    format = Documenter.HTML(;
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
    warnonly = [:missing_docs],
    # don't require match on trailing digits in floating point numbers
    doctestfilters = [r"(\d*)\.(\d{4})\d+" => s"\1.\2***"],
)
