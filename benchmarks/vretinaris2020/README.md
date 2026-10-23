# Vretinaris2020 reproduction benchmark

This benchmark refits the empirical relations registered under
`Vretinaris2020` in EmpiricalRelations.jl using the source-normalized data in
that repository. It deliberately reads the sibling repository at run time;
the source data are not copied into EmpiricalRelationsForge.jl.

From the EmpiricalRelationsForge.jl repository root, instantiate the isolated
benchmark environment and register both local packages:

```julia
julia --project=benchmarks/vretinaris2020 -e '
using Pkg
Pkg.develop([
    PackageSpec(path=pwd()),
    PackageSpec(path=joinpath(dirname(pwd()), "EmpiricalRelations.jl")),
])
Pkg.instantiate()'
```

Then reproduce all fixed-form fits and run the symbolic parameter study:

```sh
julia --project=benchmarks/vretinaris2020 benchmarks/vretinaris2020/run.jl --symbolic
```

An alternative EmpiricalRelations.jl checkout may be passed as the first
positional argument. `--quick` reduces the symbolic study to one short search
per direction for a smoke test.

The generated files in `results/` contain one row per published relation, a
symbolic-search comparison, and a concise scientific summary. The
mass-weighted tidal-deformability fit is reported as unavailable when the
normalized waveform table does not contain `lambda_tilde` values.
