"""Search-space definitions and decoded solution containers."""

from evocore.search_space.codec import (
    OperatorCodec,
    decode_gene_values,
    encode_gene_values,
    repair_gene_value,
    repair_gene_values,
)
from evocore.search_space.constraints import (
    ConstraintViolation,
    ParameterRepair,
    ParameterValidator,
    RepairRecord,
)
from evocore.search_space.genes import Gene, GeneKind, GeneSpace
from evocore.search_space.projection import (
    ActiveGeneProjection,
    ParameterProjection,
    ProjectionResult,
    ProjectionSnapshot,
)
from evocore.search_space.solutions import GeneValue, Solution, SolutionSet
from evocore.search_space.transforms import (
    BinaryThresholdTransform,
    ExponentialIntegerTransform,
    IdentityTransform,
    OutputNameTransform,
    ParameterTransform,
)

__all__ = [
    "ActiveGeneProjection",
    "BinaryThresholdTransform",
    "ConstraintViolation",
    "ExponentialIntegerTransform",
    "Gene",
    "GeneKind",
    "GeneSpace",
    "GeneValue",
    "IdentityTransform",
    "OperatorCodec",
    "OutputNameTransform",
    "ParameterProjection",
    "ParameterRepair",
    "ParameterTransform",
    "ParameterValidator",
    "ProjectionResult",
    "ProjectionSnapshot",
    "RepairRecord",
    "Solution",
    "SolutionSet",
    "decode_gene_values",
    "encode_gene_values",
    "repair_gene_value",
    "repair_gene_values",
]
