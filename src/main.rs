use verifiable_inference::{
    circuit::Circuit,
    compiler::{concat_exprs, find_max_var_id, flatten, linear_to_exprs},
    model::{Dense, Layer},
    protocol::Model,
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

fn main() {
    //let mut model = vec![];
    //for i in 0..10 {
    //    let conv = Conv {
    //        stride: 1,
    //        weight: vec![vec![vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]; 1]; 1],
    //        bias: vec![0],
    //        input_size: 100 - (i * 2),
    //    };
    //    model.push(conv);
    //}
    //
    //let input = vec![vec![vec![1; 100]; 100]];
    //let output = compute_conv(input, &model);
    //println!("{:?}", output.len());
    //println!("{:?}", output[0].len());
    //println!("{:?}", output[0][0].len());

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
