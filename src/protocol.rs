// Parties: Client, Worker, TrustedParty
//
// Model(input) -> output
// R1CS: Model(input) =?= output
//
// 1. TrustedParty
//     Model -> Circuit, R1CS
//     R1CS  -> Gens
//     R1CS, Gens -> Commitment
//
// 2. Worker
//     TrustedParty -> Circuit
//     TrustedParty -> R1CS
//     TrustedParty -> Gens
//
// 3. Client
//     TrustedParty -> Gens, Commitment
//
// 4. Worker
//     Client -> Input
//     Circuit, Input -> Output
//     R1CS, Input, Output -> Proof
//
// 5. Client
//     Worker -> Proof, Output
//     Commitment, Gens, Input, Output, Proof -> Verify

use std::rc::Rc;

use libspartan::{Assignment, ComputationCommitment, ComputationDecommitment, SNARKGens, SNARK};
use merlin::Transcript;

use crate::{
    circuit::{self, Circuit},
    model::MnistModel,
    r1cs::{into_r1cs, R1CS},
    scalar::from_i64,
};

pub trait Model {
    fn compute(&self, input: &[i64]) -> Vec<i64>;
    fn circuits(&self) -> Vec<Circuit>;
}

pub struct TrustedParty<M: Model + Clone> {
    model: M,
    circuits: Vec<Circuit>,
    r1cs: Rc<R1CS>,
    gens: Rc<SNARKGens>,
    commitment: Rc<ComputationCommitment>,
    decommitment: Rc<ComputationDecommitment>,
}

pub fn save_circuits() {
    use crate::bin_loader::save_to_file;
    let model = MnistModel::load();
    let circuits = model.circuits();
    save_to_file(&circuits, "circuits/circuits.bin").unwrap();
}

impl<M: Model + Clone> TrustedParty<M> {
    pub fn setup(model: M, circuits: Vec<Circuit>) -> Self {
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

    pub fn compute(&self, input: &[i64]) -> usize {
        self.model
            .compute(input)
            .into_iter()
            .enumerate()
            .max_by(|a, b| a.1.cmp(&b.1))
            .unwrap()
            .0
    }

    pub fn assigment_worker(&self) -> Worker<M> {
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

pub struct Worker<M: Model> {
    model: M,
    circuits: Vec<Circuit>,
    r1cs: Rc<R1CS>,
    gens: Rc<SNARKGens>,
    comm: Rc<ComputationCommitment>,
    decomm: Rc<ComputationDecommitment>,
}

impl<M: Model> Worker<M> {
    pub fn run(&self, input: &[i64]) -> (Vec<i64>, SNARK) {
        let output = self.model.compute(input);
        let inputs = [input.to_vec(), output.clone()].concat();
        let vars = crate::circuit::get_variables(&self.circuits, inputs);
        let mut transcript = Transcript::new(b"SNARK");
        // R1CS input = Circuit.input + Circuit.output
        let inputs = Assignment::new(
            &input
                .iter()
                .copied()
                .chain(output.clone())
                .map(|x| from_i64(x).to_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let vars = Assignment::new(
            &vars
                .into_iter()
                .map(|x| from_i64(x).to_bytes())
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
    pub gens: Rc<SNARKGens>,
    pub commitment: Rc<ComputationCommitment>,
}

impl Client {
    pub fn delegate_computation<M: Model>(
        &self,
        input: &[i64],
        worker: &Worker<M>,
    ) -> Option<Vec<i64>> {
        let (output, proof) = worker.run(input);
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
    use crate::{
        compiler::{find_max_var_id, flatten, linear_to_exprs},
        model::{Conv, Dense},
    };
    use libspartan::{InputsAssignment, VarsAssignment};

    #[test]
    fn test_dense_layer() {
        let dense = Dense {
            weight: vec![vec![1, 1], vec![1, 1]],
            bias: vec![1, 1],
        };
        let input = vec![1, 1];

        let exprs = linear_to_exprs(dense);
        let mut variable_counter = find_max_var_id(&exprs) + 1;
        let mut circuits = vec![];
        for expr in exprs.into_iter() {
            let (next_var, circuit) = flatten(expr, variable_counter);
            circuits.extend(circuit);
            variable_counter = next_var;
        }

        let num_vars = circuit::num_vars(&circuits);
        let variable = circuit::get_variables(&circuits, input.clone());
        let num_inputs = circuit::num_inputs(&circuits);

        let (r1cs, _) = into_r1cs(circuits.clone(), num_inputs, num_vars);
        let vars = variable
            .into_iter()
            .map(|x| from_i64(x).to_bytes())
            .collect::<Vec<_>>();
        let assignment_vars = VarsAssignment::new(&vars).unwrap();
        let inputs = input
            .into_iter()
            .chain(vec![3, 3])
            .map(|x| from_i64(x).to_bytes())
            .collect::<Vec<_>>();
        let assignment_inputs = InputsAssignment::new(&inputs).unwrap();
        assert_eq!(
            r1cs.instance.is_sat(&assignment_vars, &assignment_inputs),
            Ok(true)
        );
    }

    #[test]
    fn test_conv_layer() {
        let model = MnistModel {
            conv11: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 4,
            },
            conv12: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 4,
            },
            conv21: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 3,
            },
            conv22: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 3,
            },
            fc: Dense {
                weight: vec![
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                ],
                bias: vec![1, 1, 1, 1],
            },
        };

        let input = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let circuits = model.circuits();
        let num_vars = circuit::num_vars(&circuits);
        let variable = (0..num_vars)
            .map(|x| circuit::get_variable(&circuits, &input, x))
            .collect::<Vec<_>>();

        let num_inputs = circuit::num_inputs(&circuits);
        assert_eq!(num_inputs, 16 + 4);

        let (r1cs, _) = into_r1cs(circuits.clone(), num_inputs, num_vars);
        let vars = variable
            .into_iter()
            .map(|x| from_i64(x).to_bytes())
            .collect::<Vec<_>>();
        let assignment_vars = VarsAssignment::new(&vars).unwrap();
        let inputs = input
            .into_iter()
            .chain(vec![40805, 40805, 40805, 40805])
            .map(|x| from_i64(x).to_bytes())
            .collect::<Vec<_>>();
        let assignment_inputs = InputsAssignment::new(&inputs).unwrap();
        assert_eq!(
            r1cs.instance.is_sat(&assignment_vars, &assignment_inputs),
            Ok(true)
        );
    }
}
