use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libspartan::Assignment;
use merlin::Transcript;
use mnist::MnistBuilder;
use verifiable_inference::{
    bin_loader::load_from_file,
    circuit::Circuit,
    compiler::{concat_exprs, conv_to_exprs, find_max_var_id, flatten, linear_to_exprs},
    model::{Conv, Dense, Layer, MnistModel},
    protocol::{Model, TrustedParty},
    scalar::from_i64,
};

#[allow(dead_code)]
fn bench_mnist(c: &mut Criterion) {
    let model = MnistModel::load();
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("mnist")
        .finalize();

    let mut buffer = vec![];
    let circuits: Vec<Circuit> = load_from_file("circuits/circuits.bin", &mut buffer).unwrap();
    let trusted_party = TrustedParty::setup(model, circuits);

    let worker = trusted_party.assigment_worker();
    let client = trusted_party.create_client();

    let input = &mnist.tst_img[0..784];
    let input = input
        .iter()
        .map(|&x| if x > 0 { 1 } else { 0 })
        .collect::<Vec<i64>>();

    let (output, proof) = worker.run(&input);
    let mut transcript = Transcript::new(b"SNARK");
    let inputs = Assignment::new(
        &input
            .iter()
            .copied()
            .chain(output.clone())
            .map(|x| from_i64(x).to_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    c.bench_function("verifiable Method", |b| {
        b.iter(|| {
            proof.verify(
                black_box(&client.commitment),
                black_box(&inputs),
                black_box(&mut transcript),
                black_box(&client.gens),
            )
        })
    });
}

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

#[allow(dead_code)]
fn bench_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bench Dense: Verifiable Method");
    let input = vec![1; 100];

    for i in 1..=10 {
        let mut layers = vec![];
        for _ in 0..i {
            let dense = Dense {
                weight: vec![vec![0; 100]; 100],
                bias: vec![1; 100],
            };
            layers.push(dense);
        }

        let model = DenseModel { layers };
        let circuits = model.circuits();

        let trusted_party = TrustedParty::setup(model, circuits);

        let worker = trusted_party.assigment_worker();
        let client = trusted_party.create_client();

        let (output, proof) = worker.run(&input);
        let inputs = Assignment::new(
            &input
                .iter()
                .copied()
                .chain(output.clone())
                .map(|x| from_i64(x).to_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::new("Number of layers", i), &i, |b, &_| {
            b.iter(|| {
                proof
                    .verify(
                        black_box(&client.commitment),
                        black_box(&inputs),
                        black_box(&mut Transcript::new(b"SNARK")),
                        black_box(&client.gens),
                    )
                    .unwrap();
            })
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bench Conv: Verifiable Method");

    let size = 50;
    for num_layer in 1..=20 {
        let mut layers = vec![];

        for i in 0..num_layer {
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
        println!("Layer {num_layer}: {}", circuits.len());

        //let trusted_party = TrustedParty::setup(model, circuits);
        //
        //let worker = trusted_party.assigment_worker();
        //let client = trusted_party.create_client();
        //
        //let input = vec![1; size * size];
        //let (output, proof) = worker.run(&input);
        //let inputs = Assignment::new(
        //    &input
        //        .iter()
        //        .copied()
        //        .chain(output.clone())
        //        .map(|x| from_i64(x).to_bytes())
        //        .collect::<Vec<_>>(),
        //)
        //.unwrap();

        group.bench_with_input(
            BenchmarkId::new("Number of Layers", num_layer),
            &size,
            |b, &_| {
                b.iter(|| {
                    // dummy
                    //proof
                    //    .verify(
                    //        black_box(&client.commitment),
                    //        black_box(&inputs),
                    //        black_box(&mut Transcript::new(b"SNARK")),
                    //        black_box(&client.gens),
                    //    )
                    //    .unwrap();
                })
            },
        );
    }

    group.finish();
}

//criterion_group!(benches, bench_mnist, bench_dense);
criterion_group!(benches, bench_conv);
criterion_main!(benches);
