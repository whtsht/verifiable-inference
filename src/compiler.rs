// model
// input [a b
//        c d]
// conv  [x] <-
// conv  [y]
// output [e f
//         g h]
//
// expression
//
// a * x * y = e,
// b * x * y = f,
// c * x * y = g,
// d * x * y = h,
//
// Flattening (Algebraic Circuit)
//
// $1 = a  * x
// e  = $1 * y
// $1 = b  * x
// f  = $1 * y
// ...
//
// c * x * y = g
// d * x * y = h
//
// Our R1CS instance is three constraints over five variables and two public inputs
// (Z0 + Z1) * I0 - Z2 = 0
// (Z0 + I1) * Z2 - Z3 = 0
// Z4 * 1 - 0 = 0

use crate::model::Linear;

pub type Id = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Variable(Id),
    Constant(u32),
    Eq(Id, Box<Expression>),
    Sum(Vec<Expression>),
    Product(Vec<Expression>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Circuit {
    // x = a
    Eq(Id, CircuitValue),
    // x = a + b
    Mult(Id, CircuitValue, CircuitValue),
    // x = a * b
    Add(Id, CircuitValue, CircuitValue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitValue {
    Variable(Id),
    Constant(u32),
}

pub fn linear_to_expression(model: Linear, mut offset: Id) -> (Vec<Expression>, Id) {
    let mut expressions = vec![];
    for i in 0..model.output {
        let mut sum = vec![];
        for j in 0..model.input {
            sum.push(Expression::Product(vec![
                Expression::Variable(j + offset),
                Expression::Constant(model.weight[i * model.input + j]),
            ]));
        }
        sum.push(Expression::Constant(model.bias[i]));
        expressions.push(Expression::Eq(
            i + model.input + offset,
            Box::new(Expression::Sum(sum)),
        ));
    }
    offset += model.input + model.output;
    (expressions, offset)
}

pub fn flatten(expression: Expression, offset: &mut Id) -> Vec<Circuit> {
    match expression {
        Expression::Eq(id, expr) => match expr.as_ref() {
            Expression::Sum(exprs) => {
                let mut circuits = Vec::new();
                let mut it = exprs.iter();

                let first = match it.next() {
                    Some(Expression::Variable(id)) => CircuitValue::Variable(*id),
                    _ => todo!(),
                };
                let second = match it.next() {
                    Some(Expression::Variable(id)) => CircuitValue::Variable(*id),
                    _ => todo!(),
                };

                let mut current_value = *offset;
                circuits.push(Circuit::Add(current_value, first, second));
                *offset += 1;

                for expr in it {
                    let next_value = match expr {
                        Expression::Variable(id) => CircuitValue::Variable(*id),
                        _ => todo!(),
                    };

                    let new_id = *offset;
                    *offset += 1;

                    circuits.push(Circuit::Add(
                        new_id,
                        CircuitValue::Variable(current_value),
                        next_value,
                    ));
                    current_value = new_id;
                }

                circuits.push(Circuit::Add(
                    id,
                    CircuitValue::Variable(current_value),
                    CircuitValue::Constant(0),
                ));

                circuits
            }
            _ => todo!(),
        },
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cons(value: u32) -> Expression {
        Expression::Constant(value)
    }

    fn var(id: Id) -> Expression {
        Expression::Variable(id)
    }

    fn eq(id: Id, expr: Expression) -> Expression {
        Expression::Eq(id, Box::new(expr))
    }

    fn sum(exprs: &[Expression]) -> Expression {
        Expression::Sum(exprs.to_vec())
    }

    fn prod(exprs: &[Expression]) -> Expression {
        Expression::Product(exprs.to_vec())
    }

    #[test]
    fn test_linear_to_expression() {
        let model = Linear {
            input: 2,
            output: 1,
            weight: vec![1, 2],
            bias: vec![3],
        };
        assert_eq!(
            linear_to_expression(model, 0),
            // v2 = v0 * 1 + v1 * 2 + 3
            (
                vec![eq(
                    2,
                    sum(&[prod(&[var(0), cons(1)]), prod(&[var(1), cons(2)]), cons(3)])
                )],
                3
            )
        );

        let model = Linear {
            input: 2,
            output: 2,
            weight: vec![1, 2, 3, 4],
            bias: vec![5, 6],
        };
        assert_eq!(
            linear_to_expression(model, 0),
            (
                vec![
                    // v2 = v0 * 1 + v1 * 2 + 5
                    eq(
                        2,
                        sum(&[prod(&[var(0), cons(1)]), prod(&[var(1), cons(2)]), cons(5)])
                    ),
                    // v3 = v0 * 3 + v1 * 4 + 6
                    eq(
                        3,
                        sum(&[prod(&[var(0), cons(3)]), prod(&[var(1), cons(4)]), cons(6)])
                    ),
                ],
                4
            )
        );

        let model = Linear {
            input: 4,
            output: 4,
            weight: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            bias: vec![17, 18, 19, 20],
        };

        assert_eq!(
            linear_to_expression(model, 0),
            (
                vec![
                    // v4 = v0 * 1 + v1 * 2 + v2 * 3 + v3 * 4 + 17
                    eq(
                        4,
                        sum(&[
                            prod(&[var(0), cons(1)]),
                            prod(&[var(1), cons(2)]),
                            prod(&[var(2), cons(3)]),
                            prod(&[var(3), cons(4)]),
                            cons(17)
                        ])
                    ),
                    // v5 = v0 * 5 + v1 * 6 + v2 * 7 + v3 * 8 + 18
                    eq(
                        5,
                        sum(&[
                            prod(&[var(0), cons(5)]),
                            prod(&[var(1), cons(6)]),
                            prod(&[var(2), cons(7)]),
                            prod(&[var(3), cons(8)]),
                            cons(18)
                        ])
                    ),
                    // v6 = v0 * 9 + v1 * 10 + v2 * 11 + v3 * 12 + 19
                    eq(
                        6,
                        sum(&[
                            prod(&[var(0), cons(9)]),
                            prod(&[var(1), cons(10)]),
                            prod(&[var(2), cons(11)]),
                            prod(&[var(3), cons(12)]),
                            cons(19)
                        ])
                    ),
                    // v7 = v0 * 13 + v1 * 14 + v2 * 15 + v3 * 16 + 20
                    eq(
                        7,
                        sum(&[
                            prod(&[var(0), cons(13)]),
                            prod(&[var(1), cons(14)]),
                            prod(&[var(2), cons(15)]),
                            prod(&[var(3), cons(16)]),
                            cons(20)
                        ])
                    ),
                ],
                8
            )
        );
    }

    #[test]
    fn test_flatten() {
        // v3 = v0 + v1 + v2
        let expression = eq(3, sum(&[var(0), var(1), var(2)]));
        assert_eq!(
            flatten(expression, &mut 4),
            // g1 = v0 + v1
            // g2 = g1 + v2
            // v3 = g2 + 0
            vec![
                Circuit::Add(4, CircuitValue::Variable(0), CircuitValue::Variable(1)),
                Circuit::Add(5, CircuitValue::Variable(4), CircuitValue::Variable(2)),
                Circuit::Add(3, CircuitValue::Variable(5), CircuitValue::Constant(0)),
            ]
        );
    }
}
