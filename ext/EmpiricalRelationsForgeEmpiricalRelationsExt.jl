module EmpiricalRelationsForgeEmpiricalRelationsExt

using EmpiricalRelations
using EmpiricalRelationsForge

function EmpiricalRelationsForge.published_predictor(spec::EmpiricalRelations.RelationSpec;
                                                      strict_domain::Bool=true)
    metadata = EmpiricalRelations.info(spec)
    source = EmpiricalRelations.source(spec)
    units = Dict{Symbol,String}(metadata.quantity => string(metadata.output_unit))
    evaluator = values -> EmpiricalRelations.evaluate(spec;
        check_domain=true, strict_domain=strict_domain, values...)
    provenance = Dict{Symbol,Any}(
        :publication => metadata.publication,
        :source_id => source.source_id,
        :doi => source.doi,
        :url => source.url,
        :verification_status => metadata.verification_status,
    )
    EmpiricalRelationsForge.PublishedPredictor(string(metadata.id), metadata.inputs,
        metadata.quantity, evaluator; units,
        expression=EmpiricalRelations.formula(spec).canonical, provenance)
end

end
