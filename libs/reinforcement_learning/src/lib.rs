use std::ops::Index;

use rand::{seq::SliceRandom, Rng, RngCore};
use rand_distr::StandardNormal;

pub trait Individual {
    fn from_chromosome(chromosome: Chromosome) -> Self;
    fn as_chromosome(&self) -> &Chromosome;
    fn fitness(&self) -> f64;
}

pub enum TestIndividual {
    WithChromosome { chromosome: Chromosome },
    WithFitness { fitness: f64 },
}

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

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f64>,
}

impl Chromosome {
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl IntoIterator for Chromosome {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl FromIterator<f64> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

pub trait Selection {
    fn select<'a, I: Individual>(
        &self,
        rng: &mut dyn RngCore,
        population: &'a [I],
        cnt: u32,
    ) -> Vec<&'a I>;
}

pub struct FitnessProportionateSelection;

impl FitnessProportionateSelection {
    pub fn new() -> Self {
        Self
    }
}

impl Selection for FitnessProportionateSelection {
    fn select<'a, I: Individual>(
        &self,
        rng: &mut dyn RngCore,
        population: &'a [I],
        cnt: u32,
    ) -> Vec<&'a I> {
        assert!(!population.is_empty());

        (0..cnt)
            .map(|_| {
                population
                    .choose_weighted(rng, |chrom| chrom.fitness())
                    .unwrap()
            })
            .collect()
    }
}

pub trait Crossover {
    fn cross(
        &self,
        rng: &mut dyn RngCore,
        chromosome1: &Chromosome,
        chromosome2: &Chromosome,
    ) -> Chromosome;
}

pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl Crossover for UniformCrossover {
    fn cross(
        &self,
        rng: &mut dyn RngCore,
        chromosome1: &Chromosome,
        chromosome2: &Chromosome,
    ) -> Chromosome {
        assert!(chromosome1.len() == chromosome2.len());

        chromosome1
            .iter()
            .zip(chromosome2.iter())
            .map(|(&x, &y)| if rng.gen_bool(0.5) { x } else { y })
            .collect()
    }
}

pub trait Mutation {
    fn mutate(&self, rng: &mut dyn RngCore, chromosome: &Chromosome) -> Chromosome;
}

pub struct GaussianMutation {
    mutation_rate: f64,
    mutation_strength: f64,
}

impl GaussianMutation {
    pub fn new(mutation_rate: f64, mutation_strength: f64) -> Self {
        assert!(mutation_rate >= 0.0 && mutation_rate <= 1.0);
        Self {
            mutation_rate,
            mutation_strength,
        }
    }
}

impl Mutation for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, chromosome: &Chromosome) -> Chromosome {
        chromosome
            .iter()
            .map(|&x| {
                if rng.gen_bool(self.mutation_rate) {
                    let mutation: f64 = rng.sample(StandardNormal);
                    x + mutation * self.mutation_strength
                } else {
                    x
                }
            })
            .collect()
    }
}

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

    pub mod test_selection {
        use std::collections::BTreeMap;

        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn select_multiple() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let selector = FitnessProportionateSelection::new();
            let population = vec![
                TestIndividual::from_fitness(1.0),
                TestIndividual::from_fitness(2.0),
                TestIndividual::from_fitness(4.0),
                TestIndividual::from_fitness(0.0),
            ];

            let actual_freq: BTreeMap<i32, _> = selector
                .select(&mut rng, &population, 100)
                .iter()
                .fold(BTreeMap::new(), |mut freq, individual| {
                    *freq.entry(individual.fitness() as _).or_insert(0) += 1;
                    freq
                });

            let expected_freq = BTreeMap::from_iter([(1, 16), (2, 33), (4, 51)]);
            assert_eq!(actual_freq, expected_freq);
        }

        #[test]
        fn select_single() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let selector = FitnessProportionateSelection::new();
            let population = vec![
                TestIndividual::from_fitness(1.0),
                TestIndividual::from_fitness(2.0),
                TestIndividual::from_fitness(4.0),
                TestIndividual::from_fitness(0.0),
            ];

            let selected: Vec<Vec<&TestIndividual>> = (0..100)
                .map(|_| selector.select(&mut rng, &population, 1))
                .collect();
            let actual_freq: BTreeMap<i32, _> =
                selected
                    .iter()
                    .fold(BTreeMap::new(), |mut freq, selection| {
                        *freq.entry(selection[0].fitness() as _).or_insert(0) += 1;
                        freq
                    });

            let expected_freq = BTreeMap::from_iter([(1, 16), (2, 33), (4, 51)]);
            assert_eq!(actual_freq, expected_freq);
        }
    }

    pub mod test_crossover {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test_cross() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let crosser = UniformCrossover::new();
            let chromosome1 = Chromosome::new(vec![1.0; 50]);
            let chromosome2 = Chromosome::new(vec![-1.0; 50]);

            let actual_freq: Vec<f64> = (0..10)
                .map(|_| {
                    crosser
                        .cross(&mut rng, &chromosome1, &chromosome2)
                        .iter()
                        .sum()
                })
                .collect();

            // Sum of crossed values should be around 0
            let expected_freq = [4.0, -2.0, -12.0, 20.0, -2.0, -4.0, -2.0, 0.0, 6.0, -4.0];
            approx::assert_relative_eq!(actual_freq.as_slice(), expected_freq.as_slice());
        }

        #[test]
        #[should_panic]
        fn test_different_chromosome_length() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let crosser = UniformCrossover::new();
            let chromosome1 = Chromosome::new(vec![1.0; 2]);
            let chromosome2 = Chromosome::new(vec![-1.0; 3]);

            crosser.cross(&mut rng, &chromosome1, &chromosome2);
        }
    }

    pub mod test_mutation {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        fn mutation_result(mutation_rate: f64, mutation_strength: f64) -> Chromosome {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mutator = GaussianMutation::new(mutation_rate, mutation_strength);
            let chromosome = Chromosome::new(vec![0.0; 10]);
            mutator.mutate(&mut rng, &chromosome)
        }

        pub mod zero_mutation_rate {
            use super::*;

            #[test]
            fn zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(0.0, 0.0).into_iter().collect();
                let expected = vec![0.0; 10];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }

            #[test]
            fn non_zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(0.0, 3.0).into_iter().collect();
                let expected = vec![0.0; 10];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }

        pub mod fifty_fifty_mutation_rate {
            use super::*;

            #[test]
            fn zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(0.5, 0.0).into_iter().collect();
                let expected = vec![0.0; 10];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }

            #[test]
            fn non_zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(0.5, 3.0).into_iter().collect();
                let expected = [
                    0.0,
                    0.0,
                    -5.805140939015626,
                    -3.1938477523292375,
                    -3.00768665123635,
                    1.180657446878217,
                    0.0,
                    1.09884921578972,
                    1.83028600682312,
                    0.0,
                ];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }

        pub mod max_mutation_rate {
            use super::*;

            #[test]
            fn zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(1.0, 0.0).into_iter().collect();
                let expected = vec![0.0; 10];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }

            #[test]
            fn non_zero_mutation_strength() {
                let actual: Vec<f64> = mutation_result(1.0, 3.0).into_iter().collect();
                let expected = [
                    4.133091620252374,
                    1.2160404739410853,
                    -3.5888066801612037,
                    -5.805140939015626,
                    -2.015321177136662,
                    -3.1938477523292375,
                    -6.446928791612275,
                    -3.00768665123635,
                    -1.033546224817904,
                    1.180657446878217,
                ];
                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }

    mod test_genetic_algorithm {
        use super::*;
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
                .map(|individual| individual.as_chromosome().genes)
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
}
