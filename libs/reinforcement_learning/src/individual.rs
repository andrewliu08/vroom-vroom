use crate::chromosome::Chromosome;

pub trait Individual {
    fn from_chromosome(chromosome: Chromosome) -> Self;
    fn as_chromosome(&self) -> &Chromosome;
    fn fitness(&self) -> f64;
}

// TestIndividual used only in tests
#[allow(dead_code)]
pub enum TestIndividual {
    WithChromosome { chromosome: Chromosome },
    WithFitness { fitness: f64 },
}

#[allow(dead_code)]
impl TestIndividual {
    pub fn from_fitness(fitness: f64) -> Self {
        Self::WithFitness { fitness }
    }
}

impl Individual for TestIndividual {
    fn from_chromosome(chromosome: Chromosome) -> Self {
        Self::WithChromosome { chromosome }
    }

    fn as_chromosome(&self) -> &Chromosome {
        match self {
            Self::WithChromosome { chromosome } => chromosome,
            Self::WithFitness { .. } => panic!("Not supported for TestIndividual::WithFitness"),
        }
    }

    fn fitness(&self) -> f64 {
        match self {
            Self::WithChromosome { chromosome } => chromosome.iter().sum(),
            Self::WithFitness { fitness } => *fitness,
        }
    }
}
