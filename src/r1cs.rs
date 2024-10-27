use curve25519_dalek::Scalar;
use libspartan::Instance;

use crate::compiler::{Circuit, CircuitValue};

pub struct R1CS {
    pub num_consts: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub instance: Instance,
}

pub fn into_r1cs(circuits: Vec<Circuit>, num_inputs: usize, num_vars: usize) -> R1CS {
    let mut a: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut b: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let mut c: Vec<(usize, usize, [u8; 32])> = Vec::new();

    let num_consts = circuits.len();
    let one = Scalar::ONE.to_bytes();

    for (i, circuit) in circuits.into_iter().enumerate() {
        match circuit {
            Circuit::Eq(y, x) => {
                match x {
                    CircuitValue::Constant(con) => {
                        a.push((i, num_vars, Scalar::from(con).to_bytes()));
                    }
                    CircuitValue::Variable(var) => {
                        a.push((i, var, one));
                    }
                    CircuitValue::Input(input) => {
                        a.push((i, num_vars + 1 + input, one));
                    }
                }
                b.push((i, num_vars, one));
                c.push((i, y, one));
            }
            Circuit::Mult(y, x1, x2) => {
                match x1 {
                    CircuitValue::Constant(con) => {
                        a.push((i, num_vars, Scalar::from(con).to_bytes()));
                    }
                    CircuitValue::Variable(var) => {
                        a.push((i, var, one));
                    }
                    CircuitValue::Input(input) => {
                        a.push((i, num_vars + 1 + input, one));
                    }
                }
                match x2 {
                    CircuitValue::Constant(con) => {
                        b.push((i, num_vars, Scalar::from(con).to_bytes()));
                    }
                    CircuitValue::Variable(var) => {
                        b.push((i, var, one));
                    }
                    CircuitValue::Input(input) => {
                        b.push((i, num_vars + 1 + input, one));
                    }
                }
                c.push((i, y, one));
            }
            Circuit::Add(y, x1, x2) => {
                match x1 {
                    CircuitValue::Constant(con) => {
                        a.push((i, num_vars, Scalar::from(con).to_bytes()));
                    }
                    CircuitValue::Variable(var) => {
                        a.push((i, var, one));
                    }
                    CircuitValue::Input(input) => {
                        a.push((i, num_vars + 1 + input, one));
                    }
                }
                match x2 {
                    CircuitValue::Constant(con) => {
                        a.push((i, num_vars, Scalar::from(con).to_bytes()));
                    }
                    CircuitValue::Variable(var) => {
                        a.push((i, var, one));
                    }
                    CircuitValue::Input(input) => {
                        a.push((i, num_vars + 1 + input, one));
                    }
                }
                b.push((i, num_vars, one));
                c.push((i, y, one));
            }
        }
    }

    let instance = Instance::new(num_consts, num_vars, num_inputs, &a, &b, &c).unwrap();

    R1CS {
        num_consts,
        num_vars,
        num_inputs,
        instance,
    }
}

#[cfg(test)]
mod tests {
    use libspartan::{InputsAssignment, VarsAssignment};

    use super::*;

    #[test]
    fn test_eq_r1cs() {
        let circuits = vec![Circuit::Eq(0, CircuitValue::Constant(1234))];
        let r1cs = into_r1cs(circuits, 0, 1);
        assert_eq!(r1cs.num_consts, 1);
        assert_eq!(r1cs.num_vars, 1);
        assert_eq!(r1cs.num_inputs, 0);
        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[Scalar::from(1234u32).to_bytes()]).unwrap(),
                &InputsAssignment::new(&[]).unwrap()
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_add_r1cs() {
        // v0 = 30 + i0
        let circuits = vec![Circuit::Add(
            0,
            CircuitValue::Constant(30),
            CircuitValue::Input(0),
        )];
        let r1cs = into_r1cs(circuits, 1, 1);
        assert_eq!(r1cs.num_consts, 1);

        // v0 = 30 + 20 = 50
        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[Scalar::from(50u32).to_bytes()]).unwrap(),
                &InputsAssignment::new(&[Scalar::from(20u32).to_bytes()]).unwrap()
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_mult_r1cs() {
        // v0 = 30 * i0
        let circuits = vec![Circuit::Mult(
            0,
            CircuitValue::Constant(30),
            CircuitValue::Input(0),
        )];
        let r1cs = into_r1cs(circuits, 1, 1);
        assert_eq!(r1cs.num_consts, 1);

        // v0 = 30 * 20 = 600
        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[Scalar::from(600u32).to_bytes()]).unwrap(),
                &InputsAssignment::new(&[Scalar::from(20u32).to_bytes()]).unwrap()
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_multi_r1cs() {
        // v0 = 30 * i0
        // v1 = v0 + i1
        let circuits = vec![
            Circuit::Mult(0, CircuitValue::Constant(30), CircuitValue::Input(0)),
            Circuit::Add(1, CircuitValue::Variable(0), CircuitValue::Input(1)),
        ];

        let r1cs = into_r1cs(circuits, 2, 2);
        assert_eq!(r1cs.num_consts, 2);

        // v0 = 30 * 5 = 150
        // v1 = 150 + 6 = 156
        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[
                    Scalar::from(150u32).to_bytes(),
                    Scalar::from(156u32).to_bytes()
                ])
                .unwrap(),
                &InputsAssignment::new(&[
                    Scalar::from(5u32).to_bytes(),
                    Scalar::from(6u32).to_bytes()
                ])
                .unwrap()
            ),
            Ok(true)
        );
    }
}
