using isoPEPS
using Documenter

DocMeta.setdocmeta!(isoPEPS, :DocTestSetup, :(using isoPEPS); recursive=true)

makedocs(;
    modules=[isoPEPS],
    authors="yuqingrong",
    sitename="isoPEPS.jl",
    format=Documenter.HTML(;
        canonical="https://yuqingrong.github.io/isoPEPS.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yuqingrong/isoPEPS.jl",
    devbranch="master",
)
