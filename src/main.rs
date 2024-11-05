use verifiable_inference::{
    model::{Linear, Model},
    protocol::TrustedParty,
};

fn main() {
    let model = Model::new(vec![
        Linear {
            input: 2,
            output: 2,
            weight: vec![1, 1, 1, 1],
            bias: vec![1, 1],
        },
        Linear {
            input: 2,
            output: 2,
            weight: vec![1, 1, 1, 1],
            bias: vec![1, 1],
        },
    ]);
    let trusted_party = TrustedParty::setup(model);
    let worker = trusted_party.assigment_worker();
    let client = trusted_party.create_client();

    let input = vec![1, 1];
    let output = client.delegate_computation(input, &worker);

    assert_eq!(output, Some(vec![7, 7]));
}
