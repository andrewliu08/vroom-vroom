use rand::RngCore;

pub use crate::chromosome::Chromosome;
pub use crate::crossover::{Crossover, UniformCrossover};
pub use crate::individual::Individual;
pub use crate::mutation::{GaussianMutation, Mutation};
pub use crate::selection::{FitnessProportionateSelection, Selection};

pub struct GeneticAlgorithm<S, C, M>
where
    S: Selection,
    C: Crossover,
    M: Mutation,
{
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
}

impl<S, C, M> GeneticAlgorithm<S, C, M>
where
    S: Selection,
    C: Crossover,
    M: Mutation,
{
    pub fn new(selection_method: S, crossover_method: C, mutation_method: M) -> Self {
        Self {
            selection_method,
            crossover_method,
            mutation_method,
        }
    }

    pub fn evolve<I: Individual>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I> {
        (0..population.len())
            .map(|_| {
                let parents = self.selection_method.select(rng, population, 2);
                let child = self.crossover_method.cross(
                    rng,
                    &parents[0].as_chromosome(),
                    &parents[1].as_chromosome(),
                );
                let mutated = self.mutation_method.mutate(rng, &child);
                I::from_chromosome(mutated)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chromosome::Chromosome;
    use crate::crossover::UniformCrossover;
    use crate::individual::TestIndividual;
    use crate::mutation::GaussianMutation;
    use crate::selection::FitnessProportionateSelection;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn create_individual(genes: Vec<f64>) -> TestIndividual {
        let chromosome = Chromosome::new(genes);
        TestIndividual::WithChromosome { chromosome }
    }

    #[test]
    fn test_evolve() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let evolver = GeneticAlgorithm::new(
            FitnessProportionateSelection::new(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5, 1.0),
        );

        let mut population = vec![
            create_individual(vec![0.0; 3]),
            create_individual(vec![3.0; 3]),
            create_individual(vec![1.0, 2.0, 3.0]),
        ];
        for _ in 0..50 {
            population = evolver.evolve(&mut rng, &population);
        }

        let actual_population: Vec<Vec<f64>> = population
            .iter()
            .map(|individual| {
                individual
                    .as_chromosome()
                    .iter()
                    .map(|gene| *gene)
                    .collect()
            })
            .collect();

        // Sum of genes should get higher over time since TestIndividual's fitness
        // function is sum of genes
        let expected_population = [
            [6.345492815224679, 8.791283435771014, 4.412810778916007],
            [7.330443559227281, 9.415640416297803, 4.412810778916007],
            [8.248205437089489, 9.415640416297803, 4.080506888308995],
        ];
        for (actual_genes, expected_genes) in
            actual_population.iter().zip(expected_population.iter())
        {
            approx::assert_relative_eq!(actual_genes.as_slice(), expected_genes.as_slice());
        }
    }
}
