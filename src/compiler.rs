use crate::{
    circuit::{Circuit, CircuitValue},
    model::Linear,
};

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

            let output = Box::new(Expression::Variable(i + self.variables_counter));
            expressions.push(Expression::Eq(
                output.clone(),
                Box::new(Expression::Sum(sum)),
            ));
            expressions.push(Expression::Eq(
                output.clone(),
                Box::new(Expression::Input(model.input + i + self.inputs_counter)),
            ));
        }

        self.inputs_counter += model.input;
        self.variables_counter += model.output;
        expressions
    }

    pub fn flatten(&mut self, expression: Expression) -> Vec<Circuit> {
        let mut circuits = vec![];

        fn flatten_expr(
            compiler: &mut Compiler,
            expr: Expression,
            circuits: &mut Vec<Circuit>,
        ) -> CircuitValue {
            match expr {
                Expression::Constant(val) => CircuitValue::Constant(val),
                Expression::Variable(id) => CircuitValue::Variable(id),
                Expression::Input(id) => CircuitValue::Input(id),

                Expression::Eq(lhs, rhs) => {
                    let lhs_val = flatten_expr(compiler, *lhs, circuits);
                    let rhs_val = flatten_expr(compiler, *rhs, circuits);
                    let lhs_var = match lhs_val {
                        CircuitValue::Variable(id) => id,
                        _ => {
                            let tmp_var = compiler.variables_counter;
                            compiler.variables_counter += 1;
                            circuits.push(Circuit::Eq(tmp_var, lhs_val));
                            tmp_var
                        }
                    };
                    circuits.push(Circuit::Eq(lhs_var, rhs_val));
                    lhs_val
                }

                Expression::Sum(exprs) => {
                    let mut sum_var = None;
                    for expr in exprs {
                        let val = flatten_expr(compiler, expr, circuits);
                        sum_var = match sum_var {
                            Some(current_sum) => {
                                let tmp_var = compiler.variables_counter;
                                compiler.variables_counter += 1;
                                circuits.push(Circuit::Add(tmp_var, current_sum, val));
                                Some(CircuitValue::Variable(tmp_var))
                            }
                            None => Some(val),
                        };
                    }
                    sum_var.unwrap()
                }

                Expression::Product(exprs) => {
                    let mut prod_var = None;
                    for expr in exprs {
                        let val = flatten_expr(compiler, expr, circuits);
                        prod_var = match prod_var {
                            Some(current_prod) => {
                                let tmp_var = compiler.variables_counter;
                                compiler.variables_counter += 1;
                                circuits.push(Circuit::Mult(tmp_var, current_prod, val));
                                Some(CircuitValue::Variable(tmp_var))
                            }
                            None => Some(val),
                        };
                    }
                    prod_var.unwrap()
                }
            }
        }

        flatten_expr(self, expression, &mut circuits);

        circuits
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
            // input: i0, i1, output: i2
            // v0 = i0 * 1 + i1 * 2 + 3
            // v0 = i2
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        cons(3)
                    ])
                ),
                eq(var(0), input(2))
            ],
        );
        assert_eq!(compiler.inputs_counter, 2);
        assert_eq!(compiler.variables_counter, 1);

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
                // v0 = i2
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        cons(5)
                    ])
                ),
                eq(var(0), input(2)),
                // v1 = i0 * 3 + i1 * 4 + 6
                // v1 = i3
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(4)]),
                        cons(6)
                    ])
                ),
                eq(var(1), input(3))
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
                // v0 = i4
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
                eq(var(0), input(4)),
                // v1 = i0 * 5 + i1 * 6 + i2 * 7 + i3 * 8 + 18
                // v1 = i5
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
                eq(var(1), input(5)),
                // v2 = i0 * 9 + i1 * 10 + i2 * 11 + i3 * 12 + 19
                // v2 = i6
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
                eq(var(2), input(6)),
                // v3 = i0 * 13 + i1 * 14 + i2 * 15 + i3 * 16 + 20
                // v3 = i7
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
                eq(var(3), input(7))
            ],
        );
    }

    fn ceq(v1: usize, v2: CircuitValue) -> Circuit {
        Circuit::Eq(v1, v2)
    }

    fn cadd(v1: usize, v2: CircuitValue, v3: CircuitValue) -> Circuit {
        Circuit::Add(v1, v2, v3)
    }

    fn cmult(v1: usize, v2: CircuitValue, v3: CircuitValue) -> Circuit {
        Circuit::Mult(v1, v2, v3)
    }

    fn cvar(id: usize) -> CircuitValue {
        CircuitValue::Variable(id)
    }

    fn cinput(id: usize) -> CircuitValue {
        CircuitValue::Input(id)
    }

    fn ccons(value: u32) -> CircuitValue {
        CircuitValue::Constant(value)
    }

    #[test]
    fn test_flatten() {
        let mut compiler = Compiler::new();
        compiler.variables_counter = 10;
        // v0 = i0 * 1 + i1 * 2 + 3
        let expr = eq(
            var(0),
            sum(&[
                prod(&[input(0), cons(1)]),
                prod(&[input(1), cons(2)]),
                cons(3),
            ]),
        );
        assert_eq!(
            compiler.flatten(expr),
            // v10 = i0 * 1
            // v11 = i1 * 2
            // v12 = v10 + v11
            // v13 = v12 + 3
            // v0 = v13
            vec![
                cmult(10, cinput(0), ccons(1)),
                cmult(11, cinput(1), ccons(2)),
                cadd(12, cvar(10), cvar(11)),
                cadd(13, cvar(12), ccons(3)),
                ceq(0, cvar(13)),
            ]
        );
    }
}
