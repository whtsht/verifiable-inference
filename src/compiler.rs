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
    Input(Id),
    Eq(Box<Expression>, Box<Expression>),
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
    Input(Id),
    Constant(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Compiler {
    inputs_counter: Id,
    variables_counter: Id,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            inputs_counter: 0,
            variables_counter: 0,
        }
    }
}

impl Compiler {
    pub fn linear_to_expression(&mut self, model: Linear) -> Vec<Expression> {
        let mut expressions = vec![];
        for i in 0..model.output {
            let mut sum = vec![];
            for j in 0..model.input {
                sum.push(Expression::Product(vec![
                    Expression::Input(j + self.inputs_counter),
                    Expression::Constant(model.weight[i * model.input + j]),
                ]));
            }
            sum.push(Expression::Constant(model.bias[i]));
            expressions.push(Expression::Eq(
                Box::new(Expression::Variable(i + self.variables_counter)),
                Box::new(Expression::Sum(sum)),
            ));
        }
        self.inputs_counter += model.input;
        self.variables_counter += model.output;
        expressions
    }

    pub fn flatten(&mut self, expression: Expression) -> Vec<Circuit> {
        match expression {
            Expression::Eq(expr1, expr2) => match expr2.as_ref() {
                Expression::Sum(exprs) => {
                    let mut circuits = Vec::new();
                    let mut it = exprs.iter();

                    let first = match it.next() {
                        Some(Expression::Variable(id)) => CircuitValue::Variable(*id),
                        Some(Expression::Constant(value)) => CircuitValue::Constant(*value),
                        _ => todo!(),
                    };
                    let second = match it.next() {
                        Some(Expression::Variable(id)) => CircuitValue::Variable(*id),
                        Some(Expression::Constant(value)) => CircuitValue::Constant(*value),
                        _ => todo!(),
                    };

                    let mut current_value = self.variables_counter;
                    circuits.push(Circuit::Add(current_value, first, second));
                    self.variables_counter += 1;

                    for expr in it {
                        let next_value = match expr {
                            Expression::Variable(id) => CircuitValue::Variable(*id),
                            _ => todo!(),
                        };

                        let new_id = self.variables_counter;
                        self.variables_counter += 1;

                        circuits.push(Circuit::Add(
                            new_id,
                            CircuitValue::Variable(current_value),
                            next_value,
                        ));
                        current_value = new_id;
                    }

                    match expr1.as_ref() {
                        Expression::Variable(id) => {
                            circuits.push(Circuit::Add(
                                *id,
                                CircuitValue::Variable(current_value),
                                CircuitValue::Constant(0),
                            ));
                        }
                        _ => todo!(),
                    }

                    circuits
                }
                _ => todo!(),
            },
            _ => vec![],
        }
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

    fn eq(e1: Expression, e2: Expression) -> Expression {
        Expression::Eq(Box::new(e1), Box::new(e2))
    }

    fn sum(exprs: &[Expression]) -> Expression {
        Expression::Sum(exprs.to_vec())
    }

    fn prod(exprs: &[Expression]) -> Expression {
        Expression::Product(exprs.to_vec())
    }

    fn input(id: Id) -> Expression {
        Expression::Input(id)
    }

    #[test]
    fn test_linear_to_expression() {
        let model = Linear {
            input: 2,
            output: 1,
            weight: vec![1, 2],
            bias: vec![3],
        };
        let mut compiler = Compiler::new();
        assert_eq!(
            compiler.linear_to_expression(model),
            // v0 = i0 * 1 + i1 * 2 + 3
            vec![eq(
                var(0),
                sum(&[
                    prod(&[input(0), cons(1)]),
                    prod(&[input(1), cons(2)]),
                    cons(3)
                ])
            )],
        );

        let model = Linear {
            input: 2,
            output: 2,
            weight: vec![1, 2, 3, 4],
            bias: vec![5, 6],
        };
        let mut compiler = Compiler::new();
        assert_eq!(
            compiler.linear_to_expression(model),
            vec![
                // v0 = i0 * 1 + i1 * 2 + 5
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        cons(5)
                    ])
                ),
                // v1 = i0 * 3 + i1 * 4 + 6
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(4)]),
                        cons(6)
                    ])
                ),
            ],
        );

        let model = Linear {
            input: 4,
            output: 4,
            weight: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            bias: vec![17, 18, 19, 20],
        };
        let mut compiler = Compiler::new();

        assert_eq!(
            compiler.linear_to_expression(model),
            vec![
                // v0 = i0 * 1 + i1 * 2 + i2 * 3 + i3 * 4 + 17
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        prod(&[input(2), cons(3)]),
                        prod(&[input(3), cons(4)]),
                        cons(17)
                    ])
                ),
                // v1 = i0 * 5 + i1 * 6 + i2 * 7 + i3 * 8 + 18
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(0), cons(5)]),
                        prod(&[input(1), cons(6)]),
                        prod(&[input(2), cons(7)]),
                        prod(&[input(3), cons(8)]),
                        cons(18)
                    ])
                ),
                // v2 = i0 * 9 + i1 * 10 + i2 * 11 + i3 * 12 + 19
                eq(
                    var(2),
                    sum(&[
                        prod(&[input(0), cons(9)]),
                        prod(&[input(1), cons(10)]),
                        prod(&[input(2), cons(11)]),
                        prod(&[input(3), cons(12)]),
                        cons(19)
                    ])
                ),
                // v3 = i0 * 13 + i1 * 14 + i2 * 15 + i3 * 16 + 20
                eq(
                    var(3),
                    sum(&[
                        prod(&[input(0), cons(13)]),
                        prod(&[input(1), cons(14)]),
                        prod(&[input(2), cons(15)]),
                        prod(&[input(3), cons(16)]),
                        cons(20)
                    ])
                ),
            ],
        );
    }

    //#[test]
    //fn test_flatten() {
    //    let model = Linear {
    //        input: 2,
    //        output: 1,
    //        weight: vec![1, 2],
    //        bias: vec![3],
    //    };
    //    let mut compiler = Compiler::new();
    //    // compiler.inputs_counter = 2
    //    // compiler.variables_counter = 1
    //    // v0 = i0 * 1 + i1 * 2 + 3
    //    let expressions = compiler.linear_to_expression(model);
    //    assert_eq!(
    //        compiler.flatten(expressions[0].clone()),
    //        // v1 = i0 * 1
    //        // v2 = i1 * 2
    //        // v3 = v1 + v2
    //        // v0 = v3 + 3
    //        vec![
    //            Circuit::Add(1, CircuitValue::Variable(0), CircuitValue::Constant(0)),
    //            Circuit::Add(2, CircuitValue::Variable(1), CircuitValue::Constant(0)),
    //            Circuit::Add(3, CircuitValue::Variable(1), CircuitValue::Variable(2)),
    //            Circuit::Add(0, CircuitValue::Variable(3), CircuitValue::Constant(3)),
    //        ]
    //    );
    //}
}
