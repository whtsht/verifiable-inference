use verifiable_inference::{
    model::{Linear, Model},
    protocol::TrustedParty,
};

fn main() {
    let model = Model::new(Linear {
        input: 2,
        output: 1,
        weight: vec![1, 2],
        bias: vec![3],
    });
    let trusted_party = TrustedParty::setup(model);
    let worker = trusted_party.assigment_worker();
    let client = trusted_party.create_client();

    let input = vec![5, 6];
    let output = client.delegate_computation(input, &worker);

    // expected output: 5 * 1 + 6 * 2 + 3 = 20
    println!("Output: {:?}", output);
}
