use pyo3::prelude::*;

use crate::gene_spec::GeneKind;

pub(crate) fn parse_gene_kind(kind: &str) -> PyResult<GeneKind> {
    match kind {
        "float" => Ok(GeneKind::Float),
        "int" => Ok(GeneKind::Int),
        "bool" => Ok(GeneKind::Bool),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown gene kind: '{other}'. Valid: float, int, bool"
        ))),
    }
}

pub(crate) fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>> {
    kinds_str
        .iter()
        .map(|kind| parse_gene_kind(kind.as_str()))
        .collect()
}

pub(crate) fn repair_encoded_value(value: f64, bounds: (f64, f64), kind: &GeneKind) -> f64 {
    let (low, high) = bounds;
    match kind {
        GeneKind::Float => value.clamp(low, high),
        GeneKind::Int => value.round().clamp(low, high),
        GeneKind::Bool => {
            if value >= 0.5 {
                1.0
            } else {
                0.0
            }
        }
    }
}

pub(crate) fn repair_encoded_values(
    genes: &[f64],
    bounds: &[(f64, f64)],
    kinds: &[GeneKind],
) -> Vec<f64> {
    assert_eq!(
        genes.len(),
        bounds.len(),
        "repair_encoded_values: genes/bounds mismatch"
    );
    assert_eq!(
        genes.len(),
        kinds.len(),
        "repair_encoded_values: genes/kinds mismatch"
    );

    genes
        .iter()
        .enumerate()
        .map(|(idx, &value)| repair_encoded_value(value, bounds[idx], &kinds[idx]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_encoded_value_matches_python_contract() {
        assert_eq!(
            repair_encoded_value(99.0, (-1.0, 1.0), &GeneKind::Float),
            1.0
        );
        assert_eq!(
            repair_encoded_value(20.8, (2.0, 20.0), &GeneKind::Int),
            20.0
        );
        assert_eq!(repair_encoded_value(0.49, (0.0, 1.0), &GeneKind::Bool), 0.0);
        assert_eq!(repair_encoded_value(0.5, (0.0, 1.0), &GeneKind::Bool), 1.0);
    }

    #[test]
    fn test_repair_encoded_values_repairs_full_vector() {
        let genes = vec![99.0, 1.2, 0.8];
        let bounds = vec![(-1.0, 1.0), (2.0, 20.0), (0.0, 1.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Int, GeneKind::Bool];

        assert_eq!(
            repair_encoded_values(&genes, &bounds, &kinds),
            vec![1.0, 2.0, 1.0]
        );
    }

    #[test]
    fn test_parse_gene_kinds_reports_unknown_kind() {
        Python::initialize();

        let error = parse_gene_kinds(&["float".to_string(), "bad".to_string()])
            .expect_err("unknown kind should fail");

        assert!(error.to_string().contains("Unknown gene kind"));
    }
}
