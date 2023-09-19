use rand::RngCore;

use crate::neuron::Neuron;

#[derive(Debug)]
pub struct Layer {
    pub(crate) neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    pub fn new_random(rng: &mut dyn RngCore, nin: usize, nout: usize, bias: f64) -> Self {
        let neurons = (0..nout)
            .map(|_| Neuron::new_random(rng, nin, bias))
            .collect();
        Self { neurons }
    }

    pub fn from_weight_and_biases(
        nin: usize,
        nout: usize,
        weights: &mut dyn Iterator<Item = f64>,
    ) -> Self {
        let mut neurons = Vec::with_capacity(nout);
        for _ in 0..nout {
            neurons.push(Neuron::from_weight_and_biases(nin, weights));
        }

        Self { neurons }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_new_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let layer = Layer::new_random(&mut rng, 1, 3, 1.0);

        assert_eq!(layer.neurons.len(), 3);
        assert_eq!(layer.neurons[0].weights.len(), 1);
        let actual_weights: Vec<&[f64]> = layer
            .neurons
            .iter()
            .map(|neuron| neuron.weights.as_slice())
            .collect();
        let expected_weights: Vec<&[f64]> = vec![
            &[0.6738395137652948],
            &[0.26284898813304625],
            &[-0.5351683130665029],
        ];
        approx::assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());

        approx::assert_relative_eq!(layer.neurons[0].bias, 1.0);
    }

    #[test]
    fn test_forward() {
        let layer = Layer::new(vec![
            Neuron::new(vec![2.0, 4.0], 0.0),
            Neuron::new(vec![1.0, 2.0], 1.0),
        ]);
        let actual_output = layer.forward(&[3.0, 5.0]);
        let expected_output = vec![3.0 * 2.0 + 5.0 * 4.0 + 0.0, 3.0 * 1.0 + 5.0 * 2.0 + 1.0];
        approx::assert_relative_eq!(actual_output.as_slice(), expected_output.as_slice());
    }
}
