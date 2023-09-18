use rand::{Rng, RngCore};

use nalgebra as na;

pub struct Simulation {
    world: World,
}

pub struct World {
    animals: Vec<Animal>,
    food: Vec<Food>,
}

pub struct Animal {
    position: na::Point2<f64>,
    rotation: na::Rotation2<f64>,
    speed: f64,
    consumed: u32,
}

pub struct Food {
    position: na::Point2<f64>,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        Self {
            world: World::random(rng, num_animals, num_food),
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn move_animals(&mut self) {
        for animal in &mut self.world.animals {
            // Unit vector for default direction is (1.0, 0.0)
            let displacement = animal.rotation * na::Vector2::new(animal.speed, 0.0);
            animal.position += displacement;
            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    pub fn eat_food(&mut self, rng: &mut dyn RngCore) {
        const ANIMAL_SIZE: f64 = 0.015;
        const FOOD_SIZE: f64 = 0.005;

        for animal in &mut self.world.animals {
            for food in &mut self.world.food {
                let dist = na::distance(&animal.position, &food.position);
                if dist < ANIMAL_SIZE + FOOD_SIZE {
                    animal.consumed += 1;
                    food.position = rng.gen();
                }
            }
        }
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) {
        self.move_animals();
        self.eat_food(rng);
    }
}

impl World {
    pub fn random(rng: &mut dyn RngCore, num_animals: u8, num_food: u8) -> Self {
        let animals = (0..num_animals).map(|_| Animal::random(rng)).collect();
        let food = (0..num_food).map(|_| Food::random(rng)).collect();
        Self { animals, food }
    }

    pub fn animals(&self) -> &[Animal] {
        &self.animals
    }

    pub fn food(&self) -> &[Food] {
        &self.food
    }
}

impl Animal {
    fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.001,
            consumed: 0,
        }
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

impl Food {
    fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
        }
    }

    pub fn position(&self) -> na::Point2<f64> {
        self.position
    }
}
