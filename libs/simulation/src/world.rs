use rand::RngCore;

use crate::animal::Animal;
use crate::food::Food;

pub struct World {
    pub(crate) animals: Vec<Animal>,
    pub(crate) food: Vec<Food>,
}

impl World {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        let animals = (0..num_animals).map(|_| Animal::random(rng)).collect();
        let food = (0..num_food).map(|_| Food::new_random(rng)).collect();
        Self { animals, food }
    }

    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn food(&self) -> &[Food] {
        &self.food
    }
}
