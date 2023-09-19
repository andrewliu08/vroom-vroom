use rand::{Rng, RngCore};
use rand_distr::StandardNormal;

use super::Mutation;
use crate::chromosome::Chromosome;

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

#[cfg(test)]
mod tests {
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
