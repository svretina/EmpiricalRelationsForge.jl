using CSV
using DataFrames
using EmpiricalRelations
using EmpiricalRelationsForge
using Statistics

const SOURCE_ID = "arxiv-1910-10856"
const PUBLICATION = :Vretinaris2020
const FREQUENCIES = Dict(
    "f-peak-khz" => :f_peak_khz,
    "f-2-0-khz" => :f_2_0_khz,
    "f-spiral-khz" => :f_spiral_khz,
)
const RADII = Dict(
    "r1-2" => :r_1_2_km,
    "r1-4" => :r_1_4_km,
    "r1-6" => :r_1_6_km,
    "r1-8" => :r_1_8_km,
)

function parse_args(args)
    symbolic = "--symbolic" in args
    quick = "--quick" in args
    positional = filter(arg -> !startswith(arg, "--"), args)
    provider = isempty(positional) ? normpath(joinpath(@__DIR__, "..", "..", "..", "EmpiricalRelations.jl")) : abspath(first(positional))
    return (; provider, symbolic, quick)
end

function load_provider(provider)
    data = joinpath(provider, "data")
    isfile(joinpath(data, "waveforms.csv")) || error("waveforms.csv not found under $data")
    waveforms = CSV.read(joinpath(data, "waveforms.csv"), DataFrame; stringtype=String)
    eos = CSV.read(joinpath(data, "eos.csv"), DataFrame; stringtype=String)
    metadata = CSV.read(joinpath(data, "relations", "relations.csv"), DataFrame; stringtype=String)
    filter!(:source_id => ==(SOURCE_ID), waveforms)
    filter!(:source_id => ==(SOURCE_ID), eos)
    filter!(:source_id => ==(SOURCE_ID), metadata)
    filter!(:catalogue => ==("CFC/SPH"), waveforms)
    waveforms.eos_key = replace.(uppercase.(waveforms.eos), "SLY4" => "SLY")
    eos.eos_key = uppercase.(eos.eos)
    joined = leftjoin(waveforms, select(eos, :eos_key, values(RADII)...); on=:eos_key)
    return joined, metadata
end

function tokens(id::String)
    frequency = only(key for key in keys(FREQUENCIES) if occursin("-$key-", id))
    radius = only(key for key in keys(RADII) if occursin("-$key-", id))
    population = endswith(id, "-equal-masses") ? :equal : :all
    return frequency, radius, population
end

function select_rows(joined, frequency, radius, population, family)
    fcol, rcol = FREQUENCIES[frequency], RADII[radius]
    columns = [:id, :eos, :mchirp_msun, :q, fcol, rcol]
    data = dropmissing(select(joined, columns), [:mchirp_msun, :q, fcol, rcol])
    population == :equal && filter!(:q => q -> isapprox(q, 1.0; atol=1e-12), data)
    if family == :radius_from_frequency
        radius in ("r1-2", "r1-4") && filter!(:mchirp_msun => <(1.3), data)
        radius == "r1-8" && filter!(:mchirp_msun => >(1.3), data)
    end
    return data
end

function regression_problem(joined, spec)
    id = string(spec.id)
    frequency, radius, population = tokens(id)
    data = select_rows(joined, frequency, radius, population, spec.family)
    fcol, rcol = FREQUENCIES[frequency], RADII[radius]
    m = Float64.(data.mchirp_msun)
    f = Float64.(data[!, fcol])
    r = Float64.(data[!, rcol])
    provenance = (; row_ids=data.id, eos_ids=data.eos,
                  source_papers=fill(SOURCE_ID, nrow(data)))

    if spec.family == :frequency_from_radius
        dataset = RegressionDataset(hcat(m, r), f;
            features=[:M_chirp, :R_x], target=Symbol(fcol),
            units=Dict(:M_chirp => "Msun", :R_x => "km", Symbol(fcol) => "kHz"),
            provenance...)
        basis = BasisSet(
            PowerProductTerm(:b0),
            PowerProductTerm(:b1, Dict(:M_chirp => 1); label="M_chirp"),
            PowerProductTerm(:b2, Dict(:R_x => 1); label="R_x"),
            PowerProductTerm(:b3, Dict(:M_chirp => 2); label="M_chirp^2"),
            PowerProductTerm(:b4, Dict(:M_chirp => 1, :R_x => 1); label="M_chirp*R_x"),
            PowerProductTerm(:b5, Dict(:R_x => 2); label="R_x^2"),
        )
        method = LinearBasisRegression(basis;
            target_transform=FeatureScaleTarget(:M_chirp; power=-1),
            loss_space=:transformed, solver=:svd)
    elseif spec.family == :radius_from_frequency
        dataset = RegressionDataset(hcat(m, f), r;
            features=[:M_chirp, :f_j], target=Symbol(rcol),
            units=Dict(:M_chirp => "Msun", :f_j => "kHz", Symbol(rcol) => "km"),
            provenance...)
        basis = BasisSet(
            PowerProductTerm(:b0),
            PowerProductTerm(:b1, Dict(:M_chirp => 1); label="M_chirp"),
            PowerProductTerm(:b2, Dict(:M_chirp => -1, :f_j => 1); label="f_j/M_chirp"),
            PowerProductTerm(:b3, Dict(:M_chirp => 2); label="M_chirp^2"),
            PowerProductTerm(:b4, Dict(:f_j => 1); label="f_j"),
            PowerProductTerm(:b5, Dict(:M_chirp => -2, :f_j => 2); label="(f_j/M_chirp)^2"),
        )
        method = LinearBasisRegression(basis; loss_space=:original, solver=:svd)
    else
        error("unsupported family $(spec.family)")
    end
    return dataset, method, (; frequency, radius, population)
end

adjusted_r2(r2, n, predictors) = 1 - (1 - r2) * (n - 1) / (n - predictors - 1)

function published_metrics(metadata, id)
    row = only(eachrow(filter(:relation_id => ==(id), metadata)))
    parse_or_missing(x) = ismissing(x) || (x isa AbstractString && isempty(x)) ? missing : Float64(x)
    return (; published_adjusted_r2=parse_or_missing(row.r_squared_adjusted),
            published_max_residual=parse_or_missing(row.max_residual),
            residual_unit=row.residual_unit)
end

function fit_relations(joined, metadata)
    output = NamedTuple[]
    specs = [EmpiricalRelations.relation(row.relation_id) for row in EmpiricalRelations.relations(publication=PUBLICATION)]
    for spec in specs
        id = string(spec.id)
        if spec.family == :frequency_from_tidal_deformability
            push!(output, (; relation_id=id, family=string(spec.family), frequency="f_peak",
                radius="", population="all", status="unavailable: lambda_tilde absent",
                n=0, b0=missing, b1=missing, b2=missing, b3=missing, b4=missing, b5=missing,
                published_b0=spec.coefficient_values.b0,
                published_b1=spec.coefficient_values.b1,
                published_b2=spec.coefficient_values.b2,
                published_b3=missing, published_b4=missing, published_b5=missing,
                adjusted_r2=missing, published_adjusted_r2=published_metrics(metadata, id).published_adjusted_r2,
                max_residual=missing, published_max_residual=published_metrics(metadata, id).published_max_residual,
                residual_unit=published_metrics(metadata, id).residual_unit))
            continue
        end
        dataset, method, labels = regression_problem(joined, spec)
        model = fit(dataset, method)
        report = metrics(model, dataset)
        fit_report = if spec.family == :frequency_from_radius
            chirp_mass = dataset.X[:, 1]
            metrics(EmpiricalRelationsForge.predict(model, dataset) ./ chirp_mass, dataset.y ./ chirp_mass)
        else
            report
        end
        beta = EmpiricalRelationsForge.coefficients(model)
        published = collect(values(spec.coefficient_values))
        pm = published_metrics(metadata, id)
        r2adj = ismissing(fit_report.r_squared) ? missing : adjusted_r2(fit_report.r_squared, fit_report.n, 5)
        push!(output, (; relation_id=id, family=string(spec.family),
            frequency=labels.frequency, radius=labels.radius, population=string(labels.population), status="fitted",
            n=report.n, (Symbol("b$i") => beta[i + 1] for i in 0:5)...,
            (Symbol("published_b$i") => published[i + 1] for i in 0:5)...,
            adjusted_r2=r2adj, published_adjusted_r2=pm.published_adjusted_r2,
            max_residual=report.max_absolute_error, published_max_residual=pm.published_max_residual,
            residual_unit=pm.residual_unit))
    end
    return DataFrame(output)
end

function symbolic_dataset(dataset, family)
    m = dataset.X[:, 1]
    second = dataset.X[:, 2]
    if family == :frequency_from_radius
        X = hcat(m, second)
        y = dataset.y ./ m
        names = [:M_chirp, :R_x]
        target = :frequency_over_chirp_mass
    else
        X = hcat(m, second ./ m)
        y = dataset.y
        names = [:M_chirp, :frequency_over_chirp_mass]
        target = dataset.target_name
    end
    return RegressionDataset(X, y; features=names, target,
        row_ids=dataset.row_ids, eos_ids=dataset.eos_ids,
        source_papers=dataset.source_papers)
end

function run_symbolic(joined; quick=false)
    ids = [
        Symbol("arxiv-1910-10856-frequency-from-radius-f-peak-khz-r1-6-all-masses"),
        Symbol("arxiv-1910-10856-radius-from-frequency-r1-6-f-peak-khz-all-masses"),
    ]
    configurations = quick ? [(40, 25, 11)] : [(100, 25, 11), (250, 35, 19)]
    rows = NamedTuple[]
    for id in ids
        spec = EmpiricalRelations.relation(id)
        original, method, _ = regression_problem(joined, spec)
        fixed_model = fit(original, method)
        dataset = symbolic_dataset(original, spec.family)
        fixed_predictions = EmpiricalRelationsForge.predict(fixed_model, original)
        spec.family == :frequency_from_radius && (fixed_predictions ./= original.X[:, 1])
        fixed_report = metrics(fixed_predictions, dataset.y)
        for (iterations, complexity, seed) in configurations
            search = SymbolicSearch(binary_operators=[+, -, *],
                max_complexity=complexity, niterations=iterations,
                parallelism=:multithreading, seed=seed,
                backend_options=(; populations=12, population_size=48,
                    ncycles_per_iteration=300, parsimony=1e-5))
            result = discover(dataset, search)
            candidate = best_model(result)
            report = metrics(EmpiricalRelationsForge.predict(candidate, dataset), dataset.y)
            push!(rows, (; relation_id=string(id), iterations, max_complexity=complexity,
                seed, complexity=candidate.complexity, loss=candidate.loss,
                rmse=report.rmse, max_residual=report.max_absolute_error,
                r_squared=report.r_squared, fixed_form_rmse=fixed_report.rmse,
                fixed_form_r_squared=fixed_report.r_squared,
                formula=EmpiricalRelationsForge.formula(candidate)))
        end
    end
    return DataFrame(rows)
end

function write_summary(path, fixed, symbolic)
    fitted = filter(:status => ==("fitted"), fixed)
    coeff_errors = Float64[]
    for row in eachrow(fitted), i in 0:5
        published = row[Symbol("published_b$i")]
        push!(coeff_errors, abs(row[Symbol("b$i")] - published))
    end
    open(path, "w") do io
        println(io, "# Vretinaris2020 reproduction benchmark\n")
        println(io, "The benchmark reads only the `arxiv-1910-10856` rows from the sibling EmpiricalRelations.jl data directory and uses only the CFC/SPH catalogue used for calibration in the publication.\n")
        println(io, "## Fixed-form regression\n")
        println(io, "- Catalogue entries: $(nrow(fixed))")
        println(io, "- Refit relations: $(nrow(fitted))")
        println(io, "- Unavailable relations: $(nrow(fixed) - nrow(fitted))")
        println(io, "- Median absolute coefficient difference from the rounded published values: $(round(median(coeff_errors); sigdigits=5))")
        println(io, "- Largest absolute coefficient difference: $(round(maximum(coeff_errors); sigdigits=5))\n")
        println(io, "The detailed coefficient and residual comparison is in `fixed_form_results.csv`. The adjusted coefficient of determination uses five non-intercept predictors. The refit uses the chirp masses and frequencies printed in the publication's source table. Those values are rounded, and one SHEN radius row is absent from the archived TOV table; exact coefficient equality is therefore not expected.\n")
        println(io, "The mass-weighted tidal-deformability relation is marked unavailable because `lambda_tilde` is empty for every CFC/SPH row in `waveforms.csv`. Its formula can be evaluated through EmpiricalRelations.jl, but it cannot be independently refitted from the currently normalized data.\n")
        println(io, "## Symbolic search\n")
        if isnothing(symbolic)
            println(io, "No symbolic search was requested. Run the benchmark with `--symbolic` to generate `symbolic_results.csv`.")
        else
            println(io, "The search uses the two representative all-mass `f_peak`/`R_1.6` directions, the operator set `+`, `-`, `*`, 12 populations of 48 members, 300 cycles per iteration, and a parsimony coefficient of `1e-5`. The inverse search receives `f_peak/M_chirp` as a predictor, so the published quadratic family is expressible without an unrestricted division operator. The detailed Pareto-selected results are in `symbolic_results.csv`.\n")
            for row in eachrow(symbolic)
                println(io, "- $(row.relation_id), $(row.iterations) iterations, maximum complexity $(row.max_complexity): `$(row.formula)` (R²=$(round(row.r_squared; digits=5)), RMSE=$(round(row.rmse; sigdigits=5)); fixed-form RMSE=$(round(row.fixed_form_rmse; sigdigits=5))).")
            end
            println(io, "\nFor these seeds, increasing the budget from 100 iterations/complexity 25 to 250 iterations/complexity 35 reduced RMSE, but the improvement came from higher-order terms outside the published family. For structural recovery, a useful starting policy is therefore 100 iterations, maximum complexity 25, the three polynomial operators, the dimensionless transformed predictor, and several independent seeds. The 250/35 setting is useful as a predictive comparison. Exact recovery of the published six-term family should use a quadratic grammar or the explicit `BasisSet`; an unrestricted tree search is not an identifiability proof.")
        end
    end
end

function main(args=ARGS)
    options = parse_args(args)
    joined, metadata = load_provider(options.provider)
    output = joinpath(@__DIR__, "results")
    mkpath(output)
    fixed = fit_relations(joined, metadata)
    CSV.write(joinpath(output, "fixed_form_results.csv"), fixed)
    symbolic = options.symbolic ? run_symbolic(joined; quick=options.quick) : nothing
    !isnothing(symbolic) && CSV.write(joinpath(output, "symbolic_results.csv"), symbolic)
    write_summary(joinpath(output, "summary.md"), fixed, symbolic)
    println("Wrote $(nrow(fixed)) fixed-form results to $output")
    !isnothing(symbolic) && println("Wrote $(nrow(symbolic)) symbolic-search results to $output")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
