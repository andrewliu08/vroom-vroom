use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    pub fn new_random(rng: &mut dyn RngCore, nin: usize, bias: f64) -> Self {
        // TODO: Try techniques like He initialization, Xavier (Glorot) initialization,
        // or LeCun initialization, depending on the activation function you're using.
        // TODO: try using small non-zero value for bias when using ReLU
        // e.g. 0.01, 0.1, 1.0
        let weights: Vec<f64> = (0..nin).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        Self { weights, bias }
    }

    pub fn from_weight_and_biases(nin: usize, weights: &mut dyn Iterator<Item = f64>) -> Self {
        let bias = weights.next().expect("Not enough weights");
        let neuron_weights = (0..nin)
            .map(|_| weights.next().expect("Not enough weights"))
            .collect();

        Self {
            weights: neuron_weights,
            bias,
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());

        let dot_product: f64 = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum();
        let output = dot_product + self.bias;
        // TODO: separate activation function logic
        // ReLU
        output.max(0.0)
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
        let neuron = Neuron::new_random(&mut rng, 3, 1.0);

        assert_eq!(neuron.weights.len(), 3);
        let actual_weights = neuron.weights;
        let expected_weights = vec![0.6738395137652948, 0.26284898813304625, -0.5351683130665029];
        approx::assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());

        approx::assert_relative_eq!(neuron.bias, 1.0);
    }

    #[test]
    fn test_forward() {
        let neuron = Neuron::new(vec![2.0, 4.0], 0.1);
        let actual_output = neuron.forward(&[0.5, 0.5]);
        let expected_output = 1.0 + 2.0 + 0.1;
        approx::assert_relative_eq!(actual_output, expected_output);
    }

    #[test]
    fn test_relu() {
        let neuron = Neuron::new(vec![0.5], 0.0);
        approx::assert_relative_eq!(neuron.forward(&[-3.0]), 0.0);
        approx::assert_relative_eq!(neuron.forward(&[0.0]), 0.0);
        approx::assert_relative_eq!(neuron.forward(&[4.0]), 2.0);
    }

    #[test]
    #[should_panic]
    fn test_forward_wrong_input_size() {
        let neuron = Neuron::new(vec![2.0, 4.0], 0.1);
        neuron.forward(&[0.5]);
    }
}
