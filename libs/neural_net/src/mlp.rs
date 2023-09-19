use rand::RngCore;

use crate::layer::Layer;

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn new_random(rng: &mut dyn RngCore, mut nin: usize, nouts: &[usize], bias: f64) -> Self {
        let layers = nouts
            .iter()
            .map(|&nout| {
                let layer = Layer::new_random(rng, nin, nout, bias);
                nin = nout;
                layer
            })
            .collect();
        Self { layers }
    }

    pub fn from_weight_and_biases(
        mut nin: usize,
        nouts: &[usize],
        weights: impl IntoIterator<Item = f64>,
    ) -> Self {
        let mut weights = weights.into_iter();

        let mut layers = Vec::with_capacity(nouts.len());
        for nout in nouts {
            layers.push(Layer::from_weight_and_biases(nin, *nout, &mut weights));
            nin = *nout;
        }

        Self { layers }
    }

    pub fn forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.forward(&inputs))
    }

    pub fn weights_and_biases(&self) -> Vec<f64> {
        let mut weights = Vec::new();

        for layer in &self.layers {
            for neuron in &layer.neurons {
                weights.push(neuron.bias);

                for weight in &neuron.weights {
                    weights.push(*weight);
                }
            }
        }

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::Neuron;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_new_random() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let mlp = MLP::new_random(&mut rng, 1, &[3, 2], 1.0);

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
