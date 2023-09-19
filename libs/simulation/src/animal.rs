use nalgebra as na;
use rand::{Rng, RngCore};

use lib_neural_net as nn;
use lib_reinforcement_learning::genetic_algorithm as ga;

use crate::eye::Eye;

pub struct Animal {
    pub(crate) position: na::Point2<f64>,
    pub(crate) rotation: na::Rotation2<f64>,
    pub(crate) speed: f64,
    pub(crate) consumed: u32,
    pub(crate) eye: Eye,
    pub(crate) brain: nn::MLP,
}

pub struct AnimalIndividual {
    pub(crate) chromosome: ga::Chromosome,
    pub(crate) fitness: f64,
}

impl Animal {
    pub fn new(rng: &mut dyn RngCore, eye: Eye, brain: nn::MLP) -> Self {
        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.001,
            consumed: 0,
            eye,
            brain,
        }
    }

    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();
        let brain = nn::MLP::new_random(rng, eye.receptors, &[2 * eye.receptors, 2], 0.01);
        Self::new(rng, eye, brain)
    }

    pub fn from_chromosome(rng: &mut dyn RngCore, chromosome: ga::Chromosome) -> Self {
        let eye = Eye::default();
        let brain =
            nn::MLP::from_weight_and_biases(eye.receptors, &[2 * eye.receptors, 2], chromosome);
        Self::new(rng, eye, brain)
    }

    pub fn as_chromosome(&self) -> ga::Chromosome {
        ga::Chromosome::new(self.brain.weights_and_biases())
    }

    pub fn position(&self) -> na::Point2<f64> {
        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f64> {
        self.rotation
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }
}

impl AnimalIndividual {
    pub fn from_animal(animal: &Animal) -> Self {
        Self {
            chromosome: animal.as_chromosome(),
            fitness: animal.consumed as f64,
        }
    }

    pub fn into_animal(&self, rng: &mut dyn RngCore) -> Animal {
        Animal::from_chromosome(rng, self.chromosome.clone())
    }
}

impl ga::Individual for AnimalIndividual {
    fn from_chromosome(chromosome: ga::Chromosome) -> Self {
        Self {
            chromosome,
            fitness: 0.0,
        }
    }

    fn as_chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }
}
