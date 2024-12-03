use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mnist::MnistBuilder;
use verifiable_inference::model::{Conv, Dense, Layer, MnistModel};
use verifiable_inference::protocol::Model;

#[allow(dead_code)]
fn bench_mnist(c: &mut Criterion) {
    let model = MnistModel::load();
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("mnist")
        .finalize();

    let input = &mnist.tst_img[0..784];
    let input = input
        .iter()
        .map(|&x| if x > 0 { 1 } else { 0 })
        .collect::<Vec<i64>>();

    c.bench_function("Bench Mnist: Normal Method", |b| {
        b.iter(|| model.compute(black_box(&input)))
    });
}

fn compute(input: &[i64], model: &[Dense]) -> Vec<i64> {
    let mut output = input.to_vec();
    for d in model {
        output = d.forward(output);
    }
    output
}

#[allow(dead_code)]
fn bench_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bench Dense: Normal Method");
    for i in 1..=10 {
        group.bench_with_input(BenchmarkId::new("Number of layers", i), &i, |b, &i| {
            let mut model = vec![];
            for _ in 0..i {
                let dense = Dense {
                    weight: vec![vec![1; 100]; 100],
                    bias: vec![1; 100],
                };
                model.push(dense);
            }
            let input = vec![1; 100];

            b.iter(|| compute(black_box(&input), black_box(&model)))
        });
    }
    group.finish();
}

fn compute_conv(input: Vec<Vec<Vec<i64>>>, model: &[Conv]) -> Vec<Vec<Vec<i64>>> {
    let mut output = input;
    for d in model {
        output = d.forward(output);
    }
    output
}

#[allow(dead_code)]
fn bench_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bench Conv: Normal Method");
    for size in (1..=20).map(|x| x * 10) {
        group.bench_with_input(
            BenchmarkId::new("Input image size", size),
            &size,
            |b, &size| {
                let mut model = vec![];
                for i in 0..2 {
                    let conv = Conv {
                        stride: 1,
                        weight: vec![vec![vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]; 1]; 1],
                        bias: vec![0],
                        input_size: size - (i * 2),
                    };
                    model.push(conv);
                }

                b.iter(|| {
                    compute_conv(
                        black_box(vec![vec![vec![1; size]; size]]),
                        black_box(&model),
                    )
                })
            },
        );
    }
    group.finish();
}

//criterion_group!(benches, bench_mnist, bench_dense);
criterion_group!(benches, bench_conv);
criterion_main!(benches);
