using PartitionedDistributions
using Documenter

DocMeta.setdocmeta!(PartitionedDistributions, :DocTestSetup, :(using PartitionedDistributions); recursive = true)

makedocs(;
    modules = [PartitionedDistributions],
    authors = "Seth Axen <seth@sethaxen.com> and contributors",
    sitename = "PartitionedDistributions.jl",
    format = Documenter.HTML(;
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)
