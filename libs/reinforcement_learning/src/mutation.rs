pub use self::gaussian_mutation::GaussianMutation;

use rand::RngCore;

use crate::chromosome::Chromosome;

mod gaussian_mutation;

pub trait Mutation {
    fn mutate(&self, rng: &mut dyn RngCore, chromosome: &Chromosome) -> Chromosome;
}
