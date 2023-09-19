use rand::{seq::SliceRandom, RngCore};

use super::Selection;
use crate::individual::Individual;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::individual::TestIndividual;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::BTreeMap;

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
