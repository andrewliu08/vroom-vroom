pub use self::uniform_crossover::UniformCrossover;

use rand::RngCore;

use crate::chromosome::Chromosome;

mod uniform_crossover;

pub trait Crossover {
    fn cross(
        &self,
        rng: &mut dyn RngCore,
        chromosome1: &Chromosome,
        chromosome2: &Chromosome,
    ) -> Chromosome;
}
