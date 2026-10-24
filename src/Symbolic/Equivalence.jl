struct EquivalenceAssessment
    level::Symbol
    equivalent::Bool
    maximum_absolute_difference::Float64
    message::String
end

function _canonical_signature(node, options; include_constants::Bool=false)
    if node.degree == 0
        return node.constant ? (include_constants ? "C($(repr(node.val)))" : "P") : "X$(node.feature)"
    end
    op = options.operators.ops[node.degree][node.op]
    name = string(nameof(op))
    children = [_canonical_signature(child, options; include_constants) for child in _node_children(node)]
    node.degree == 2 && (op === (+) || op === (*)) && sort!(children)
    "$name($(join(children, ",")))"
end

structural_fingerprint(candidate::SymbolicCandidate; include_constants::Bool=false) =
    _digest(_canonical_signature(_raw_tree(candidate.expression), candidate.options; include_constants))

function equivalence(a::SymbolicCandidate, b::SymbolicCandidate;
                     X=nothing, atol::Real=1e-10, rtol::Real=1e-8)
    a.feature_names == b.feature_names || return EquivalenceAssessment(:different_variables, false, Inf,
        "feature identities or order differ")
    exact_a = _tree_signature(_raw_tree(a.expression), a.options; constants=true)
    exact_b = _tree_signature(_raw_tree(b.expression), b.options; constants=true)
    exact_a == exact_b && return EquivalenceAssessment(:identical_tree, true, 0.0, "identical operator tree and constants")
    structural_fingerprint(a) == structural_fingerprint(b) &&
        return EquivalenceAssessment(:canonical_structure, true, NaN,
            "same safely canonicalized parameterized structure")
    isnothing(X) && return EquivalenceAssessment(:different_structure, false, NaN,
        "no numerical domain was supplied")
    pa, pb = predict(a, X), predict(b, X)
    difference = maximum(abs.(pa .- pb))
    similar = all(isapprox.(pa, pb; atol, rtol))
    EquivalenceAssessment(:numerical_sample, similar, difference,
        similar ? "numerically similar on supplied samples; not a proof" : "predictions differ on supplied samples")
end

function deduplicate_candidates(candidates::AbstractVector{<:SymbolicCandidate})
    groups = Dict{String,Vector{Int}}()
    for (index, candidate) in enumerate(candidates)
        push!(get!(groups, structural_fingerprint(candidate), Int[]), index)
    end
    representatives = [indices[argmin(candidates[index].loss for index in indices)]
                       for indices in values(groups)]
    (; representatives=sort(representatives), groups=sort(collect(values(groups)); by=first),
       originals=collect(candidates))
end
