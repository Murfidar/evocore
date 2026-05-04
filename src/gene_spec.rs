#[derive(Clone, Debug, PartialEq)]
pub enum GeneKind {
    Float,
    Int,
    Bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_kind_clone() {
        let k = GeneKind::Float;
        let k2 = k.clone();
        assert!(matches!(k2, GeneKind::Float));
    }

    #[test]
    fn test_gene_kind_all_variants_distinct() {
        let variants = [GeneKind::Float, GeneKind::Int, GeneKind::Bool];
        assert!(!matches!(variants[0], GeneKind::Int));
        assert!(!matches!(variants[1], GeneKind::Bool));
        assert!(!matches!(variants[2], GeneKind::Float));
    }
}
