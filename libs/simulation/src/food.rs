use nalgebra as na;
use rand::{Rng, RngCore};

pub struct Food {
    pub(crate) position: na::Point2<f64>,
}

impl Food {
    pub fn new(position: na::Point2<f64>) -> Self {
        Self { position }
    }

    pub fn new_random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
        }
    }

    pub fn randomize_position(&mut self, rng: &mut dyn RngCore) {
        self.position = rng.gen();
    }

    pub fn position(&self) -> na::Point2<f64> {
        self.position
    }
}
