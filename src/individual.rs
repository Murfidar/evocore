use pyo3::prelude::*;

#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct FloatIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<f64>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl FloatIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<f64>, fitness: Option<f64>) -> Self {
        Self { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "FloatIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct IntegerIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<i64>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl IntegerIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<i64>, fitness: Option<f64>) -> Self {
        Self { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "IntegerIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct BinaryIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<bool>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl BinaryIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<bool>, fitness: Option<f64>) -> Self {
        Self { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BinaryIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_individual_len() {
        let ind = FloatIndividual {
            genes: vec![1.0, 2.0, 3.0],
            fitness: None,
        };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_float_individual_fitness_none_default() {
        let ind = FloatIndividual {
            genes: vec![0.0],
            fitness: None,
        };
        assert!(ind.fitness.is_none());
    }

    #[test]
    fn test_float_individual_clone_preserves_fitness() {
        let ind = FloatIndividual {
            genes: vec![1.0, 2.0],
            fitness: Some(3.14),
        };
        let cloned = ind.clone();
        assert_eq!(cloned.fitness, Some(3.14));
        assert_eq!(cloned.genes, vec![1.0, 2.0]);
    }

    #[test]
    fn test_integer_individual_len() {
        let ind = IntegerIndividual {
            genes: vec![10, 20, 30],
            fitness: None,
        };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_integer_individual_clone() {
        let ind = IntegerIndividual {
            genes: vec![5, -3],
            fitness: Some(1.0),
        };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![5, -3]);
        assert_eq!(cloned.fitness, Some(1.0));
    }

    #[test]
    fn test_binary_individual_len() {
        let ind = BinaryIndividual {
            genes: vec![true, false, true],
            fitness: None,
        };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_binary_individual_clone() {
        let ind = BinaryIndividual {
            genes: vec![true, false],
            fitness: Some(2.0),
        };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![true, false]);
    }

    #[test]
    fn test_float_individual_has_no_fitness_valid_field() {
        let ind = FloatIndividual {
            genes: vec![1.0],
            fitness: None,
        };
        let _ = ind.fitness;
    }
}
