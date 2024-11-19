use crate::{
    circuit::{Circuit, CircuitValue},
    model::{Conv, Linear},
};

pub type Id = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Variable(Id),
    Constant(i32),
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

    pub fn new_with_counters(inputs_counter: Id, variables_counter: Id) -> Self {
        Compiler {
            inputs_counter,
            variables_counter,
        }
    }
}

impl Compiler {
    pub fn linear_to_exprs(&mut self, model: Linear) -> Vec<Expression> {
        let mut exprs = vec![];
        for i in 0..model.output {
            let mut sum = vec![];
            for j in 0..model.input {
                sum.push(Expression::Product(vec![
                    Expression::Input(j + self.inputs_counter),
                    Expression::Constant(model.weight[i][j]),
                ]));
            }
            sum.push(Expression::Constant(model.bias[i]));

            let output = Box::new(Expression::Variable(i + self.variables_counter));
            exprs.push(Expression::Eq(
                output.clone(),
                Box::new(Expression::Sum(sum)),
            ));
            exprs.push(Expression::Eq(
                output.clone(),
                Box::new(Expression::Input(model.input + i + self.inputs_counter)),
            ));
        }

        self.inputs_counter += model.input;
        self.variables_counter += model.output;
        exprs
    }

    pub fn conv_to_exprs(&mut self, model: Conv) -> Vec<Expression> {
        let mut exprs = vec![];

        let mut var_counter = 0;
        for (out_channel, bias) in model.weight.into_iter().zip(model.bias) {
            for kernel in out_channel.iter() {
                let kernel_size = kernel.len();
                for i in (0..=((model.input_size - kernel_size) / model.stride))
                    .map(|x| x * model.stride)
                {
                    for j in (0..=((model.input_size - kernel_size) / model.stride))
                        .map(|x| x * model.stride)
                    {
                        let expr = self.kernel_to_expr(
                            kernel.clone(),
                            self.gen_conv_input(
                                i * model.input_size + j,
                                kernel_size,
                                model.input_size,
                            ),
                            bias,
                            var_counter,
                        );
                        exprs.push(expr);
                        var_counter += 1;
                    }
                }
            }
        }

        for j in 0..var_counter {
            exprs.push(Expression::Eq(
                Box::new(Expression::Variable(j)),
                Box::new(Expression::Input(model.input_size.pow(2) + j)),
            ));
        }

        exprs
    }

    fn gen_conv_input(&self, start: usize, kernel_size: usize, img_size: usize) -> Vec<Vec<usize>> {
        (0..kernel_size)
            .map(|i| (0..kernel_size).map(|j| start + i * img_size + j).collect())
            .collect()
    }

    fn kernel_to_expr(
        &mut self,
        kernel: Vec<Vec<i32>>,
        inputs: Vec<Vec<usize>>,
        bias: i32,
        var: usize,
    ) -> Expression {
        let mut exprs = vec![];
        for (k_row, i_row) in kernel.into_iter().zip(inputs.into_iter()) {
            for (k, i) in k_row.into_iter().zip(i_row.into_iter()) {
                exprs.push(Expression::Product(vec![
                    Expression::Input(i),
                    Expression::Constant(k),
                ]));
            }
        }
        exprs.push(Expression::Constant(bias));

        Expression::Eq(
            Box::new(Expression::Variable(var)),
            Box::new(Expression::Sum(exprs)),
        )
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

pub fn remove_output(exprs: &[Expression]) -> Vec<Expression> {
    exprs
        .iter()
        .filter(|expr| match expr {
            Expression::Eq(var, input) => !matches!(
                (var.as_ref(), input.as_ref()),
                (&Expression::Variable(_), &Expression::Input(_))
            ),
            _ => true,
        })
        .cloned()
        .collect()
}

pub fn replace_var_expr(expr: Expression, from: Id, to: Id) -> Expression {
    match expr {
        Expression::Variable(id) if id == from => Expression::Variable(to),
        Expression::Eq(lhs, rhs) => Expression::Eq(
            Box::new(replace_var_expr(*lhs, from, to)),
            Box::new(replace_var_expr(*rhs, from, to)),
        ),
        Expression::Sum(exprs) => Expression::Sum(
            exprs
                .into_iter()
                .map(|expr| replace_var_expr(expr, from, to))
                .collect(),
        ),
        Expression::Product(exprs) => Expression::Product(
            exprs
                .into_iter()
                .map(|expr| replace_var_expr(expr, from, to))
                .collect(),
        ),
        expr => expr,
    }
}

pub fn replace_var(exprs: Vec<Expression>, from: Id, to: Id) -> Vec<Expression> {
    exprs
        .into_iter()
        .map(|expr| replace_var_expr(expr, from, to))
        .collect()
}

pub fn replace_input2var(expr: Expression, target: Id) -> Expression {
    match expr {
        Expression::Input(id) if id == target => Expression::Variable(target),
        Expression::Eq(lhs, rhs) => Expression::Eq(
            Box::new(replace_input2var(*lhs, target)),
            Box::new(replace_input2var(*rhs, target)),
        ),
        Expression::Sum(exprs) => Expression::Sum(
            exprs
                .into_iter()
                .map(|expr| replace_input2var(expr, target))
                .collect(),
        ),
        Expression::Product(exprs) => Expression::Product(
            exprs
                .into_iter()
                .map(|expr| replace_input2var(expr, target))
                .collect(),
        ),
        expr => expr,
    }
}

pub fn replace_input2var_all(exprs: Vec<Expression>, target: Id) -> Vec<Expression> {
    exprs
        .into_iter()
        .map(|expr| replace_input2var(expr, target))
        .collect()
}

pub fn find_max_id(exprs: &[Expression]) -> Id {
    exprs
        .iter()
        .flat_map(|expr| match expr {
            Expression::Variable(id) => vec![*id],
            Expression::Eq(a, b) => {
                let mut ids = vec![];
                if let Expression::Variable(id) = **a {
                    ids.push(id);
                }
                if let Expression::Variable(id) = **b {
                    ids.push(id);
                }
                ids
            }
            Expression::Sum(exprs) | Expression::Product(exprs) => vec![find_max_id(exprs)],
            Expression::Input(_) | Expression::Constant(_) => vec![],
        })
        .max()
        .unwrap_or(0)
}

pub fn find_max_input_id(exprs: &[Expression]) -> Id {
    exprs
        .iter()
        .flat_map(|expr| match expr {
            Expression::Input(id) => vec![*id],
            Expression::Eq(a, b) => {
                let mut ids = vec![];
                if let Expression::Input(id) = **a {
                    ids.push(id);
                }
                if let Expression::Input(id) = **b {
                    ids.push(id);
                }
                ids
            }
            Expression::Sum(exprs) | Expression::Product(exprs) => vec![find_max_input_id(exprs)],
            Expression::Variable(_) | Expression::Constant(_) => vec![],
        })
        .max()
        .unwrap_or(0)
}
pub fn concat_exprs(exprs1: Vec<Expression>, mut exprs2: Vec<Expression>) -> Vec<Expression> {
    if exprs1.is_empty() {
        return exprs2;
    }
    let max_id = find_max_id(&exprs1);
    let exprs1 = remove_output(&exprs1);
    for (old_id, new_id) in (0..=max_id).zip(max_id + 1..) {
        exprs2 = replace_var(exprs2, old_id, new_id);
    }
    for target in 0..=max_id {
        exprs2 = replace_input2var_all(exprs2, target);
    }
    exprs1.into_iter().chain(exprs2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cons(value: i32) -> Expression {
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
            weight: vec![vec![1, 2]],
            bias: vec![3],
        };
        let mut compiler = Compiler::new();
        assert_eq!(
            compiler.linear_to_exprs(model),
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
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let mut compiler = Compiler::new();
        assert_eq!(
            compiler.linear_to_exprs(model),
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
            weight: vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ],
            bias: vec![17, 18, 19, 20],
        };
        let mut compiler = Compiler::new();

        assert_eq!(
            compiler.linear_to_exprs(model),
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

    fn ccons(value: i32) -> CircuitValue {
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

    #[test]
    fn test_remove_output() {
        let model = Linear {
            input: 2,
            output: 2,
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let mut compiler = Compiler::new();

        let exprs = compiler.linear_to_exprs(model);
        assert_eq!(
            remove_output(&exprs),
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        cons(5)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(4)]),
                        cons(6)
                    ])
                ),
            ]
        )
    }

    #[test]
    fn test_replace_var() {
        let expr = eq(var(0), sum(&[var(1), var(2)]));
        assert_eq!(
            replace_var(vec![expr.clone()], 0, 3),
            vec![eq(var(3), sum(&[var(1), var(2)]),)]
        );
        assert_eq!(
            replace_var(vec![expr.clone()], 1, 3),
            vec![eq(var(0), sum(&[var(3), var(2)]),)]
        );
        assert_eq!(
            replace_var(vec![expr.clone()], 2, 3),
            vec![eq(var(0), sum(&[var(1), var(3)]),)]
        );
    }

    #[test]
    fn test_multi_layer() {
        let layer1 = Linear {
            input: 1,
            output: 1,
            weight: vec![vec![1]],
            bias: vec![1],
        };
        let layer2 = Linear {
            input: 1,
            output: 1,
            weight: vec![vec![1]],
            bias: vec![1],
        };

        let mut compiler = Compiler::new();
        let exprs1 = compiler.linear_to_exprs(layer1);
        let mut compiler = Compiler::new();
        let exprs2 = compiler.linear_to_exprs(layer2);
        assert_eq!(
            concat_exprs(exprs1, exprs2),
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(1)])),
                eq(var(1), sum(&[prod(&[var(0), cons(1)]), cons(1)])),
                eq(var(1), input(1))
            ],
        );

        let layer1 = Linear {
            input: 2,
            output: 2,
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let layer2 = Linear {
            input: 2,
            output: 2,
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let mut compiler = Compiler::new();
        let exprs1 = compiler.linear_to_exprs(layer1);
        let mut compiler = Compiler::new();
        let exprs2 = compiler.linear_to_exprs(layer2);
        assert_eq!(
            concat_exprs(exprs1, exprs2),
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        cons(5)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(4)]),
                        cons(6)
                    ])
                ),
                eq(
                    var(2),
                    sum(&[prod(&[var(0), cons(1)]), prod(&[var(1), cons(2)]), cons(5)])
                ),
                eq(var(2), input(2)),
                eq(
                    var(3),
                    sum(&[prod(&[var(0), cons(3)]), prod(&[var(1), cons(4)]), cons(6)])
                ),
                eq(var(3), input(3))
            ]
        );
    }

    #[test]
    fn test_gen_conv_input() {
        let compiler = Compiler::new();
        assert_eq!(
            compiler.gen_conv_input(0, 2, 3),
            vec![vec![0, 1], vec![3, 4]]
        );
        assert_eq!(
            compiler.gen_conv_input(1, 2, 3),
            vec![vec![1, 2], vec![4, 5]]
        );
        assert_eq!(
            compiler.gen_conv_input(3, 2, 3),
            vec![vec![3, 4], vec![6, 7]]
        );
        assert_eq!(
            compiler.gen_conv_input(4, 2, 3),
            vec![vec![4, 5], vec![7, 8]]
        );
    }

    #[test]
    fn test_conv() {
        let conv = Conv {
            input_size: 1,
            stride: 1,
            weight: vec![vec![vec![vec![12]]]],
            bias: vec![7],
        };

        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                // input: i0, output: i1
                // v0 = i0 * 1 + 1
                // v0 = i1
                eq(var(0), sum(&[prod(&[input(0), cons(12)]), cons(7)]),),
                eq(var(0), input(1)),
            ]
        );
    }

    #[test]
    fn test_conv_multi_kernel() {
        let conv = Conv {
            input_size: 3,
            stride: 1,
            weight: vec![vec![vec![vec![1, 2], vec![3, 4]]]],
            bias: vec![10],
        };

        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        prod(&[input(3), cons(3)]),
                        prod(&[input(4), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(1), cons(1)]),
                        prod(&[input(2), cons(2)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(5), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(2),
                    sum(&[
                        prod(&[input(3), cons(1)]),
                        prod(&[input(4), cons(2)]),
                        prod(&[input(6), cons(3)]),
                        prod(&[input(7), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(3),
                    sum(&[
                        prod(&[input(4), cons(1)]),
                        prod(&[input(5), cons(2)]),
                        prod(&[input(7), cons(3)]),
                        prod(&[input(8), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(var(0), input(9)),
                eq(var(1), input(10)),
                eq(var(2), input(11)),
                eq(var(3), input(12)),
            ]
        );
    }

    #[test]
    fn test_conv_stride() {
        let conv = Conv {
            input_size: 4,
            stride: 2,
            weight: vec![vec![vec![vec![1, 2], vec![3, 4]]]],
            bias: vec![10],
        };
        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(5), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(2), cons(1)]),
                        prod(&[input(3), cons(2)]),
                        prod(&[input(6), cons(3)]),
                        prod(&[input(7), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(2),
                    sum(&[
                        prod(&[input(8), cons(1)]),
                        prod(&[input(9), cons(2)]),
                        prod(&[input(12), cons(3)]),
                        prod(&[input(13), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(3),
                    sum(&[
                        prod(&[input(10), cons(1)]),
                        prod(&[input(11), cons(2)]),
                        prod(&[input(14), cons(3)]),
                        prod(&[input(15), cons(4)]),
                        cons(10)
                    ])
                ),
                eq(var(0), input(16)),
                eq(var(1), input(17)),
                eq(var(2), input(18)),
                eq(var(3), input(19)),
            ]
        );

        let conv = Conv {
            input_size: 4,
            stride: 2,
            weight: vec![vec![vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]]],
            bias: vec![10],
        };
        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        prod(&[input(2), cons(3)]),
                        prod(&[input(4), cons(4)]),
                        prod(&[input(5), cons(5)]),
                        prod(&[input(6), cons(6)]),
                        prod(&[input(8), cons(7)]),
                        prod(&[input(9), cons(8)]),
                        prod(&[input(10), cons(9)]),
                        cons(10)
                    ])
                ),
                eq(var(0), input(16)),
            ]
        );

        let conv = Conv {
            input_size: 4,
            stride: 1,
            weight: vec![vec![vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]]],
            bias: vec![10],
        };
        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(1)]),
                        prod(&[input(1), cons(2)]),
                        prod(&[input(2), cons(3)]),
                        prod(&[input(4), cons(4)]),
                        prod(&[input(5), cons(5)]),
                        prod(&[input(6), cons(6)]),
                        prod(&[input(8), cons(7)]),
                        prod(&[input(9), cons(8)]),
                        prod(&[input(10), cons(9)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(1), cons(1)]),
                        prod(&[input(2), cons(2)]),
                        prod(&[input(3), cons(3)]),
                        prod(&[input(5), cons(4)]),
                        prod(&[input(6), cons(5)]),
                        prod(&[input(7), cons(6)]),
                        prod(&[input(9), cons(7)]),
                        prod(&[input(10), cons(8)]),
                        prod(&[input(11), cons(9)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(2),
                    sum(&[
                        prod(&[input(4), cons(1)]),
                        prod(&[input(5), cons(2)]),
                        prod(&[input(6), cons(3)]),
                        prod(&[input(8), cons(4)]),
                        prod(&[input(9), cons(5)]),
                        prod(&[input(10), cons(6)]),
                        prod(&[input(12), cons(7)]),
                        prod(&[input(13), cons(8)]),
                        prod(&[input(14), cons(9)]),
                        cons(10)
                    ])
                ),
                eq(
                    var(3),
                    sum(&[
                        prod(&[input(5), cons(1)]),
                        prod(&[input(6), cons(2)]),
                        prod(&[input(7), cons(3)]),
                        prod(&[input(9), cons(4)]),
                        prod(&[input(10), cons(5)]),
                        prod(&[input(11), cons(6)]),
                        prod(&[input(13), cons(7)]),
                        prod(&[input(14), cons(8)]),
                        prod(&[input(15), cons(9)]),
                        cons(10)
                    ])
                ),
                eq(var(0), input(16)),
                eq(var(1), input(17)),
                eq(var(2), input(18)),
                eq(var(3), input(19)),
            ]
        );
    }

    #[test]
    fn test_conv_multi_channel() {
        // multi input
        let conv = Conv {
            input_size: 1,
            stride: 1,
            weight: vec![vec![vec![vec![1]], vec![vec![2]]]],
            bias: vec![3, 4],
        };
        let mut compiler = Compiler::new();
        let exprs = compiler.conv_to_exprs(conv);
        assert_eq!(
            exprs,
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(3)])),
                eq(var(0), input(1)),
                eq(var(1), sum(&[prod(&[input(0), cons(2)]), cons(4)])),
                eq(var(1), input(2)),
            ]
        );
    }
}
