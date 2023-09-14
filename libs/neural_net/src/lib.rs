use rand::{Rng, RngCore};

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl MLP {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn random_weights(
        rng: &mut dyn RngCore,
        mut nin: usize,
        nouts: &[usize],
        bias: f64,
    ) -> Self {
        let layers = nouts
            .iter()
            .map(|&nout| {
                let layer = Layer::random_weights(rng, nin, nout, bias);
                nin = nout;
                layer
            })
            .collect();
        Self { layers }
    }

    pub fn forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.forward(&inputs))
    }
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    pub fn random_weights(rng: &mut dyn RngCore, nin: usize, nout: usize, bias: f64) -> Self {
        let neurons = (0..nout)
            .map(|_| Neuron::random_weights(rng, nin, bias))
            .collect();
        Self { neurons }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    pub fn random_weights(rng: &mut dyn RngCore, nin: usize, bias: f64) -> Self {
        // TODO: Try techniques like He initialization, Xavier (Glorot) initialization,
        // or LeCun initialization, depending on the activation function you're using.
        // TODO: try using small non-zero value for bias when using ReLU
        // e.g. 0.01, 0.1, 1.0
        let weights: Vec<f64> = (0..nin).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        Self { weights, bias }
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

    mod test_neuron {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test_random_weights() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random_weights(&mut rng, 3, 1.0);

            assert_eq!(neuron.weights.len(), 3);
            let actual_weights = neuron.weights;
            let expected_weights =
                vec![0.6738395137652948, 0.26284898813304625, -0.5351683130665029];
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

    mod test_layer {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test_random_weights() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random_weights(&mut rng, 1, 3, 1.0);

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

    mod test_mlp {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test_random_weights() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let mlp = MLP::random_weights(&mut rng, 1, &[3, 2], 1.0);

            let layer0 = &mlp.layers[0];
            assert_eq!(layer0.neurons.len(), 3);
            assert_eq!(layer0.neurons[0].weights.len(), 1);
            let layer0_actual_weights: Vec<&[f64]> = layer0
                .neurons
                .iter()
                .map(|neuron| neuron.weights.as_slice())
                .collect();
            let layer0_expected_weights: Vec<&[f64]> = vec![
                &[0.6738395137652948],
                &[0.26284898813304625],
                &[-0.5351683130665029],
            ];
            approx::assert_relative_eq!(
                layer0_actual_weights.as_slice(),
                layer0_expected_weights.as_slice()
            );
            approx::assert_relative_eq!(layer0.neurons[0].bias, 1.0);

            let layer1 = &mlp.layers[1];
            assert_eq!(layer1.neurons.len(), 2);
            assert_eq!(layer1.neurons[0].weights.len(), 3);
            let layer1_actual_weights: Vec<&[f64]> = layer1
                .neurons
                .iter()
                .map(|neuron| neuron.weights.as_slice())
                .collect();
            let layer1_expected_weights: Vec<&[f64]> = vec![
                &[
                    -0.7648179607770014,
                    -0.48879602856526627,
                    -0.8020499621501127,
                ],
                &[
                    -0.9868003303940736,
                    -0.4766220977890224,
                    -0.3612778288989301,
                ],
            ];
            approx::assert_relative_eq!(
                layer1_actual_weights.as_slice(),
                layer1_expected_weights.as_slice()
            );
            approx::assert_relative_eq!(layer1.neurons[0].bias, 1.0);
        }

        #[test]
        fn test_forward() {
            let layer0 = Layer::new(vec![
                Neuron::new(vec![2.0, 4.0], 0.0),
                Neuron::new(vec![1.0, 2.0], 1.0),
            ]);
            let layer1 = Layer::new(vec![Neuron::new(vec![0.5, -0.5], 0.1)]);
            let mlp = MLP::new(vec![layer0, layer1]);

            let actual_output = mlp.forward(vec![3.0, 5.0]);
            // layer0 output: [26.0, 14.0]
            // layer1 ouput: [6.1]
            let expected_output = vec![6.1];
            approx::assert_relative_eq!(actual_output.as_slice(), expected_output.as_slice());
        }
    }
}
