/// Parties: Client, Worker, TrustedParty
///
/// Model(input) -> output
/// R1CS: Model(input) =?= output
///
/// 1. TrustedParty
///     Model -> Circuit, R1CS
///     R1CS  -> Gens
///     R1CS, Gens -> Commitment
///
/// 2. Worker
///     TrustedParty -> Circuit
///     TrustedParty -> R1CS
///     TrustedParty -> Gens
///
/// 3. Client
///     TrustedParty -> Gens, Commitment
///
/// 4. Worker
///     Client -> Input
///     Circuit, Input -> Output
///     R1CS, Input, Output -> Proof
///
/// 5. Client
///     Worker -> Proof, Output
///     Commitment, Gens, Input, Output, Proof -> Verify
use std::rc::Rc;

use libspartan::{Assignment, ComputationCommitment, ComputationDecommitment, SNARKGens, SNARK};
use merlin::Transcript;

use crate::{
    circuit::{self, Circuit},
    model::Model,
    r1cs::{into_r1cs, R1CS},
    scalar::from_i32,
};

pub struct TrustedParty {
    model: Model,
    circuits: Vec<Circuit>,
    r1cs: Rc<R1CS>,
    gens: Rc<SNARKGens>,
    commitment: Rc<ComputationCommitment>,
    decommitment: Rc<ComputationDecommitment>,
}

impl TrustedParty {
    pub fn setup(model: Model) -> Self {
        let circuits = model.clone().circuits();
        let num_cons = circuits.len();
        let num_vars = circuit::num_vars(&circuits);
        let num_inputs = circuit::num_inputs(&circuits);

        let (r1cs, num_non_zero_entries) = into_r1cs(circuits.clone(), num_inputs, num_vars);
        let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_non_zero_entries);

        let (comm, decomm) = SNARK::encode(&r1cs.instance, &gens);
        Self {
            model,
            circuits,
            r1cs: Rc::new(r1cs),
            gens: Rc::new(gens),
            commitment: Rc::new(comm),
            decommitment: Rc::new(decomm),
        }
    }

    pub fn assigment_worker(&self) -> Worker {
        Worker {
            model: self.model.clone(),
            circuits: self.circuits.clone(),
            r1cs: Rc::clone(&self.r1cs),
            gens: Rc::clone(&self.gens),
            comm: Rc::clone(&self.commitment),
            decomm: Rc::clone(&self.decommitment),
        }
    }

    pub fn create_client(&self) -> Client {
        Client {
            gens: Rc::clone(&self.gens),
            commitment: Rc::clone(&self.commitment),
        }
    }
}

pub struct Worker {
    model: Model,
    circuits: Vec<Circuit>,
    r1cs: Rc<R1CS>,
    gens: Rc<SNARKGens>,
    comm: Rc<ComputationCommitment>,
    decomm: Rc<ComputationDecommitment>,
}

impl Worker {
    pub fn run(&self, input: Vec<i32>) -> (Vec<i32>, SNARK) {
        let output = self.model.compute(&input);
        let inputs = [input.clone(), output.clone()].concat();
        let vars = crate::circuit::get_variables(&self.circuits, inputs);
        let mut transcript = Transcript::new(b"SNARK");
        // R1CS input = Circuit.input + Circuit.output
        let inputs = Assignment::new(
            &input
                .into_iter()
                .chain(output.clone())
                .map(|x| from_i32(x).to_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let vars = Assignment::new(
            &vars
                .into_iter()
                .map(|x| from_i32(x).to_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let proof = SNARK::prove(
            &self.r1cs.instance,
            &self.comm,
            &self.decomm,
            vars,
            &inputs,
            &self.gens,
            &mut transcript,
        );
        (output, proof)
    }
}

pub struct Client {
    gens: Rc<SNARKGens>,
    commitment: Rc<ComputationCommitment>,
}

impl Client {
    pub fn delegate_computation(&self, input: Vec<i32>, worker: &Worker) -> Option<Vec<i32>> {
        let (output, proof) = worker.run(input.clone());
        let mut transcript = Transcript::new(b"SNARK");
        let inputs = Assignment::new(
            &input
                .into_iter()
                .chain(output.clone())
                .map(|x| from_i32(x).to_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        match proof.verify(&self.commitment, &inputs, &mut transcript, &self.gens) {
            Ok(()) => Some(output),
            Err(err) => {
                println!("Error: {:?}", err);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Linear;

    #[test]
    fn test_protorol() {
        let model = Model::new(vec![
            Linear {
                input: 2,
                output: 2,
                weight: vec![vec![1, 1], vec![1, 1]],
                bias: vec![1, 1],
            },
            Linear {
                input: 2,
                output: 2,
                weight: vec![vec![1, 1], vec![1, 1]],
                bias: vec![1, -9],
            },
        ]);
        let trusted_party = TrustedParty::setup(model);
        let worker = trusted_party.assigment_worker();
        let client = trusted_party.create_client();

        let input = vec![1, 1];
        let output = client.delegate_computation(input, &worker);

        assert_eq!(output, Some(vec![7, -3]));
    }
}
