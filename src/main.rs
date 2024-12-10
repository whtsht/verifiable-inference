use verifiable_inference::{
    circuit::Circuit,
    compiler::{concat_exprs, conv_to_exprs, find_max_var_id, flatten, linear_to_exprs},
    model::{Conv, Dense, Layer},
    protocol::{Model, TrustedParty},
};

#[derive(Clone)]
pub struct DenseModel {
    layers: Vec<Dense>,
}

impl Model for DenseModel {
    fn compute(&self, input: &[i64]) -> Vec<i64> {
        let mut output = input.to_vec();
        for d in self.layers.iter() {
            output = d.forward(output);
        }
        output
    }

    fn circuits(&self) -> Vec<Circuit> {
        let mut exprs = vec![];

        for dense in self.layers.iter() {
            let new_exprs = linear_to_exprs(dense.clone());
            exprs = concat_exprs(exprs, new_exprs);
        }

        let mut variable_counter = find_max_var_id(&exprs) + 1;
        let mut circuits = vec![];
        for expr in exprs {
            let (next_var, circuit) = flatten(expr, variable_counter);
            circuits.extend(circuit);
            variable_counter = next_var;
        }

        circuits
    }
}

#[derive(Clone)]
struct ConvModel {
    layers: Vec<Conv>,
    input_size: usize,
}

impl Model for ConvModel {
    fn compute(&self, input: &[i64]) -> Vec<i64> {
        let mut output: Vec<Vec<Vec<i64>>> = vec![input
            .chunks(self.input_size)
            .map(|row| row.to_vec())
            .collect()];

        for c in self.layers.iter() {
            output = c.forward(output);
        }
        output.into_iter().flatten().flatten().collect()
    }

    fn circuits(&self) -> Vec<Circuit> {
        let mut exprs = vec![];

        for conv in self.layers.iter() {
            let new_exprs = conv_to_exprs(conv.clone());
            exprs = concat_exprs(exprs, new_exprs);
        }

        let mut variable_counter = find_max_var_id(&exprs) + 1;
        let mut circuits = vec![];
        for expr in exprs {
            let (next_var, circuit) = flatten(expr, variable_counter);
            circuits.extend(circuit);
            variable_counter = next_var;
        }

        circuits
    }
}

fn main() {
    let mut layers = vec![];
    let size = 28;
    for i in 0..10 {
        let conv = Conv {
            stride: 1,
            weight: vec![vec![vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]]],
            bias: vec![0],
            input_size: size - (i * 2),
        };
        layers.push(conv);
    }

    let model = ConvModel {
        layers,
        input_size: size,
    };

    let circuits = model.circuits();
    println!("{}", circuits.len());
    let trusted_party = TrustedParty::setup(model, circuits);

    let worker = trusted_party.assigment_worker();
    let client = trusted_party.create_client();
    let input = vec![1; size * size];
    println!(
        "{:?}",
        client.delegate_computation(&input, &worker).is_some()
    );

    //let model = Model::load();
    //let mnist = MnistBuilder::new()
    //    .label_format_digit()
    //    .base_path("mnist")
    //    .finalize();
    //
    //let trusted_party = TrustedParty::setup(model);
    //
    //let worker = trusted_party.assigment_worker();
    //let client = trusted_party.create_client();
    //
    //let idx = 100;
    //let input = &mnist.tst_img[784 * idx..784 * idx + (28 * 28)];
    //let input = input
    //    .iter()
    //    .map(|&x| if x > 0 { 1 } else { 0 })
    //    .collect::<Vec<u8>>();
    //let output = client.delegate_computation(&input, &worker).unwrap();
    //let (label, _) = output
    //    .into_iter()
    //    .enumerate()
    //    .max_by(|a, b| a.1.cmp(&b.1))
    //    .unwrap();
    //
    //assert_eq!(label, trusted_party.compute(&input));
}
