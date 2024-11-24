use mnist::MnistBuilder;
use verifiable_inference::{model::Model, protocol::TrustedParty};

//fn save_circuits() {
//    use verifiable_inference::bin_loader::save_to_file;
//    let model = Model::load();
//    let circuits = model.circuits();
//    println!("circuits.len() = {}", circuits.len());
//    save_to_file(&circuits, "circuits.bin").unwrap();
//}

fn main() {
    //save_circuits();
    let model = Model::load();
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .base_path("mnist")
        .finalize();

    let idx = 0;
    let input = &mnist.tst_img[784 * idx..784 * idx + (28 * 28)];

    let input = input
        .iter()
        .map(|&x| if x > 0 { 1 } else { 0 })
        .collect::<Vec<u8>>();
    // Debug print
    //for ds in input.chunks(28) {
    //    for d in ds {
    //        if d != &0 {
    //            print!("1")
    //        } else {
    //            print!("0")
    //        }
    //    }
    //    println!()
    //}

    //println!(
    //    "label {} : predicted_label {:?}",
    //    mnist.tst_lbl[idx],
    //    model.compute(&input)
    //);

    let trusted_party = TrustedParty::setup(model);
    let worker = trusted_party.assigment_worker();
    let client = trusted_party.create_client();
    let output = client.delegate_computation(input, &worker).unwrap();
    let (label, _) = output
        .into_iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(&b.1))
        .unwrap();
    assert_eq!(label as u8, mnist.tst_lbl[idx]);

    //let accuracy = calculate_accuracy(&model, &mnist);
    //println!("accuracy: {:.2}%", accuracy);
}

//fn calculate_accuracy(model: &Model, mnist: &Mnist) -> f64 {
//    let mut correct_predictions = 0;
//    let total_samples = mnist.tst_lbl.len();
//
//    for (idx, label) in mnist.tst_lbl.iter().enumerate() {
//        let input = &mnist.tst_img[784 * idx..784 * idx + (28 * 28)];
//        let input = input
//            .iter()
//            .map(|&x| if x > 0 { 1 } else { 0 })
//            .collect::<Vec<u8>>();
//
//        let (predicted_label, _) = model
//            .compute(&input)
//            .iter()
//            .enumerate()
//            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//            .unwrap();
//
//        if predicted_label as u8 == *label {
//            correct_predictions += 1;
//        }
//    }
//
//    correct_predictions as f64 / total_samples as f64 * 100.0
//}
