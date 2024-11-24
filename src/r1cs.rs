use curve25519_dalek::Scalar;
use libspartan::Instance;

use crate::{
    circuit::{Circuit, CircuitValue},
    scalar::from_i64,
};

pub struct R1CS {
    pub num_consts: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub instance: Instance,
}

pub fn into_r1cs(circuits: Vec<Circuit>, num_inputs: usize, num_vars: usize) -> (R1CS, usize) {
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
                        a.push((i, num_vars, from_i64(con).to_bytes()));
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
                        a.push((i, num_vars, from_i64(con).to_bytes()));
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
                        b.push((i, num_vars, from_i64(con).to_bytes()));
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
                        a.push((i, num_vars, from_i64(con).to_bytes()));
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
                        a.push((i, num_vars, from_i64(con).to_bytes()));
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

    let num_non_zero_entries = a.len().max(b.len()).max(c.len());

    (
        R1CS {
            num_consts,
            num_vars,
            num_inputs,
            instance,
        },
        num_non_zero_entries,
    )
}

#[cfg(test)]
mod tests {
    use libspartan::{InputsAssignment, VarsAssignment};

    use super::*;

    #[test]
    fn test_eq_r1cs() {
        let circuits = vec![Circuit::Eq(0, CircuitValue::Constant(1234))];
        let (r1cs, _) = into_r1cs(circuits, 0, 1);
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
        let (r1cs, _) = into_r1cs(circuits, 1, 1);
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
        let (r1cs, _) = into_r1cs(circuits, 1, 1);
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

        let (r1cs, _) = into_r1cs(circuits, 2, 2);
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

    #[test]
    fn test_r1cs_with_output() {
        // v0 = 30 + i0
        // v0 = i1
        let circuits = vec![
            Circuit::Add(0, CircuitValue::Constant(30), CircuitValue::Input(0)),
            Circuit::Eq(0, CircuitValue::Input(1)),
        ];
        let (r1cs, _) = into_r1cs(circuits, 2, 1);
        assert_eq!(r1cs.num_consts, 2);

        // v0 = 30 + 20 = 50
        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[Scalar::from(50u32).to_bytes()]).unwrap(),
                &InputsAssignment::new(&[
                    Scalar::from(20u32).to_bytes(),
                    Scalar::from(50u32).to_bytes()
                ])
                .unwrap()
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_multi_r1cs_with_output() {
        // v0 = 30 * i0
        // v1 = v0 + i1
        // v0 = i2
        // v1 = i3
        let circuits = vec![
            Circuit::Mult(0, CircuitValue::Constant(30), CircuitValue::Input(0)),
            Circuit::Add(1, CircuitValue::Variable(0), CircuitValue::Input(1)),
            Circuit::Eq(0, CircuitValue::Input(2)),
            Circuit::Eq(1, CircuitValue::Input(3)),
        ];

        let (r1cs, _) = into_r1cs(circuits, 4, 2);
        assert_eq!(r1cs.num_consts, 4);

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
                    Scalar::from(6u32).to_bytes(),
                    Scalar::from(150u32).to_bytes(),
                    Scalar::from(156u32).to_bytes()
                ])
                .unwrap()
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_r1cs() {
        // v0 = i0 + 1
        // v1 = i1 + 1
        // v2 = v0 * v1
        // v2 = i2
        let circuits = vec![
            Circuit::Add(0, CircuitValue::Input(0), CircuitValue::Constant(1)),
            Circuit::Add(1, CircuitValue::Input(1), CircuitValue::Constant(1)),
            Circuit::Mult(2, CircuitValue::Variable(0), CircuitValue::Variable(1)),
            Circuit::Eq(2, CircuitValue::Input(2)),
        ];
        let (r1cs, _) = into_r1cs(circuits, 3, 3);
        assert_eq!(r1cs.num_consts, 4);

        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(&[
                    Scalar::from(2u32).to_bytes(),
                    Scalar::from(2u32).to_bytes(),
                    Scalar::from(4u32).to_bytes(),
                ])
                .unwrap(),
                &InputsAssignment::new(&[
                    Scalar::from(1u32).to_bytes(),
                    Scalar::from(1u32).to_bytes(),
                    Scalar::from(4u32).to_bytes(),
                ])
                .unwrap()
            ),
            Ok(true)
        );
    }

    //#[test]
    //fn test_dense_layer_r1cs() {
    //    let circuits = vec![
    //        Circuit::Mult(1, CircuitValue::Input(0), CircuitValue::Constant(1)),
    //        Circuit::Mult(2, CircuitValue::Input(1), CircuitValue::Constant(1)),
    //        Circuit::Add(3, CircuitValue::Variable(1), CircuitValue::Variable(2)),
    //        Circuit::Add(4, CircuitValue::Variable(3), CircuitValue::Constant(1)),
    //        Circuit::Eq(0, CircuitValue::Variable(4)),
    //        Circuit::Eq(0, CircuitValue::Input(2)),
    //        Circuit::Mult(5, CircuitValue::Input(0), CircuitValue::Constant(1)),
    //        Circuit::Mult(6, CircuitValue::Input(1), CircuitValue::Constant(1)),
    //        Circuit::Add(7, CircuitValue::Variable(5), CircuitValue::Variable(6)),
    //        Circuit::Add(8, CircuitValue::Variable(7), CircuitValue::Constant(1)),
    //        Circuit::Eq(1, CircuitValue::Variable(8)),
    //        Circuit::Eq(1, CircuitValue::Input(3)),
    //    ];
    //
    //    let (r1cs, _) = into_r1cs(circuits, 4, 9);
    //    assert_eq!(r1cs.num_consts, 12);
    //
    //    assert_eq!(
    //        r1cs.instance.is_sat(
    //            &VarsAssignment::new(
    //                &[3u32, 1, 1, 2, 3, 1, 1, 2, 3]
    //                    .into_iter()
    //                    .map(|x| Scalar::from(x).to_bytes())
    //                    .collect::<Vec<_>>()
    //            )
    //            .unwrap(),
    //            &InputsAssignment::new(&[
    //                Scalar::from(1u32).to_bytes(),
    //                Scalar::from(1u32).to_bytes(),
    //                Scalar::from(3u32).to_bytes(),
    //                Scalar::from(3u32).to_bytes(),
    //            ])
    //            .unwrap()
    //        ),
    //        Ok(true)
    //    );
    //}

    #[test]
    fn test_dense_layer_r1cs() {
        let circuits = vec![
            Circuit::Mult(1, CircuitValue::Input(0), CircuitValue::Constant(1)),
            Circuit::Mult(2, CircuitValue::Input(1), CircuitValue::Constant(1)),
            Circuit::Add(3, CircuitValue::Variable(1), CircuitValue::Variable(2)),
            Circuit::Add(4, CircuitValue::Variable(3), CircuitValue::Constant(1)),
            Circuit::Eq(0, CircuitValue::Variable(4)),
            Circuit::Eq(0, CircuitValue::Input(2)),
        ];

        let (r1cs, _) = into_r1cs(circuits, 3, 5);
        assert_eq!(r1cs.num_consts, 6);

        assert_eq!(
            r1cs.instance.is_sat(
                &VarsAssignment::new(
                    &[3u32, 1, 1, 2, 3]
                        .into_iter()
                        .map(|x| Scalar::from(x).to_bytes())
                        .collect::<Vec<_>>()
                )
                .unwrap(),
                &InputsAssignment::new(&[
                    Scalar::from(1u32).to_bytes(),
                    Scalar::from(1u32).to_bytes(),
                    Scalar::from(3u32).to_bytes(),
                ])
                .unwrap()
            ),
            Ok(true)
        );
    }
}
