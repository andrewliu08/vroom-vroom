use rand::{Rng, RngCore};

use super::Crossover;
use crate::chromosome::Chromosome;

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

#[cfg(test)]
mod tests {
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
