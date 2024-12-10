use crate::{
    circuit::{Circuit, CircuitValue},
    model::{Conv, Dense},
};
use rayon::prelude::*;

pub type Id = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Variable(Id),
    Constant(i64),
    Input(Id),
    Eq(Box<Expression>, Box<Expression>),
    Sum(Vec<Expression>),
    Product(Vec<Expression>),
}

pub fn get_variables(exprs: &[Expression], input: Vec<i64>) -> Vec<i64> {
    let mut vars = vec![0i64; get_max_id(exprs) + 1];

    for expr in exprs {
        if let Expression::Eq(a, b) = expr {
            if let &Expression::Variable(id) = a.as_ref() {
                vars[id] = evaluate_expr(*b.clone(), &input, &vars);
            }
        }
    }

    vars
}

fn get_max_id(exprs: &[Expression]) -> Id {
    exprs
        .par_iter()
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
            Expression::Sum(exprs) | Expression::Product(exprs) => vec![get_max_id(exprs)],
            Expression::Input(_) | Expression::Constant(_) => vec![],
        })
        .max()
        .unwrap_or(0)
}

pub fn max_input_id(exprs: &Vec<Expression>) -> usize {
    let mut max_id = 0;
    for expr in exprs {
        match expr {
            Expression::Variable(_) => {}
            Expression::Constant(_) => {}
            Expression::Input(i) => {
                max_id = max_id.max(*i);
            }
            Expression::Eq(e1, e2) => {
                max_id = max_id.max(max_input_id_(e1));
                max_id = max_id.max(max_input_id_(e2));
            }
            Expression::Sum(vec) | Expression::Product(vec) => {
                max_id = max_id.max(max_input_id(vec));
            }
        }
    }

    max_id
}

fn max_input_id_(expr: &Expression) -> usize {
    let mut max_id = 0;
    match expr {
        Expression::Variable(_) => {}
        Expression::Constant(_) => {}
        Expression::Input(i) => {
            max_id = max_id.max(*i);
        }
        Expression::Eq(e1, e2) => {
            max_id = max_id.max(max_input_id_(e1));
            max_id = max_id.max(max_input_id_(e2));
        }
        Expression::Sum(vec) | Expression::Product(vec) => {
            max_id = max_id.max(max_input_id(vec));
        }
    }
    max_id
}

fn evaluate_expr(expr: Expression, input: &[i64], vars: &[i64]) -> i64 {
    match expr {
        Expression::Constant(val) => val,
        Expression::Variable(id) => vars[id],
        Expression::Input(id) => input[id],
        Expression::Sum(exprs) => exprs
            .into_iter()
            .map(|expr| evaluate_expr(expr, input, vars))
            .sum(),
        Expression::Product(exprs) => exprs
            .into_iter()
            .map(|expr| evaluate_expr(expr, input, vars))
            .product(),
        Expression::Eq(_, _) => unreachable!(),
    }
}

pub fn linear_to_exprs(model: Dense) -> Vec<Expression> {
    let output_len = model.weight.len();
    let input_len = model.weight[0].len();

    let mut exprs = vec![];
    for i in 0..output_len {
        let mut sum = vec![];
        for j in 0..input_len {
            sum.push(Expression::Product(vec![
                Expression::Input(j),
                Expression::Constant(model.weight[i][j]),
            ]));
        }
        sum.push(Expression::Constant(model.bias[i]));

        let output = Box::new(Expression::Variable(i));
        exprs.push(Expression::Eq(
            output.clone(),
            Box::new(Expression::Sum(sum)),
        ));
        exprs.push(Expression::Eq(
            output.clone(),
            Box::new(Expression::Input(input_len + i)),
        ));
    }

    exprs
}

pub fn conv_to_exprs(model: Conv) -> Vec<Expression> {
    let mut exprs = vec![];

    let mut var_counter = 0;
    let input_channel_size = model.weight[0].len();
    let kernel_size = model.weight[0][0].len();
    let input_size = model.input_size;

    for (out_channel, bias) in model.weight.into_iter().zip(model.bias) {
        let output_size = (input_size - kernel_size) / model.stride + 1;
        let mut input_kernel_exprs: Vec<Vec<Vec<Expression>>> =
            vec![vec![vec![]; output_size]; output_size];

        for (input_idx, kernel) in out_channel.iter().enumerate() {
            for i in (0..output_size).map(|x| x * model.stride) {
                for j in (0..output_size).map(|x| x * model.stride) {
                    let inputs = gen_conv_input(
                        input_idx * input_size * input_size + i * input_size + j,
                        kernel_size,
                        input_size,
                    );

                    let mut tmp_exprs = vec![];
                    for (k_row, i_row) in kernel.clone().into_iter().zip(inputs.into_iter()) {
                        for (k, i) in k_row.into_iter().zip(i_row.into_iter()) {
                            tmp_exprs.push(Expression::Product(vec![
                                Expression::Input(i),
                                Expression::Constant(k),
                            ]));
                        }
                    }

                    input_kernel_exprs[i / model.stride][j / model.stride].extend(tmp_exprs);
                }
            }
        }

        for kernel_exprs_row in input_kernel_exprs {
            for mut kernel_exprs in kernel_exprs_row {
                kernel_exprs.push(Expression::Constant(bias));

                let expr = Expression::Eq(
                    Box::new(Expression::Variable(var_counter)),
                    Box::new(Expression::Sum(kernel_exprs)),
                );

                exprs.push(expr);
                var_counter += 1;
            }
        }
    }

    for j in 0..var_counter {
        exprs.push(Expression::Eq(
            Box::new(Expression::Variable(j)),
            Box::new(Expression::Input(
                input_size.pow(2) * input_channel_size + j,
            )),
        ));
    }

    exprs
}

fn gen_conv_input(start: usize, kernel_size: usize, img_size: usize) -> Vec<Vec<usize>> {
    (0..kernel_size)
        .map(|i| (0..kernel_size).map(|j| start + i * img_size + j).collect())
        .collect()
}

pub fn flatten(expression: Expression, variables_counter: Id) -> (Id, Vec<Circuit>) {
    fn flatten_expr(
        mut variables_counter: Id,
        expr: Expression,
    ) -> (CircuitValue, Id, Vec<Circuit>) {
        match expr {
            Expression::Constant(val) => (CircuitValue::Constant(val), variables_counter, vec![]),
            Expression::Variable(id) => (CircuitValue::Variable(id), variables_counter, vec![]),
            Expression::Input(id) => (CircuitValue::Input(id), variables_counter, vec![]),
            Expression::Eq(lhs, rhs) => {
                let (lhs_val, lhs_counter, lhs_circuits) = flatten_expr(variables_counter, *lhs);
                let (rhs_val, rhs_counter, rhs_circuits) = flatten_expr(lhs_counter, *rhs);
                variables_counter = rhs_counter;

                let mut circuits = lhs_circuits;
                circuits.extend(rhs_circuits);

                let lhs_var = match lhs_val {
                    CircuitValue::Variable(id) => id,
                    _ => {
                        let tmp_var = variables_counter;
                        variables_counter += 1;
                        circuits.push(Circuit::Eq(tmp_var, lhs_val));
                        tmp_var
                    }
                };

                circuits.push(Circuit::Eq(lhs_var, rhs_val));
                (lhs_val, variables_counter, circuits)
            }

            Expression::Sum(exprs) => {
                let mut sum_var = None;
                let mut circuits = vec![];

                for expr in exprs {
                    let (val, next_counter, new_circuits) = flatten_expr(variables_counter, expr);
                    variables_counter = next_counter;
                    circuits.extend(new_circuits);

                    sum_var = match sum_var {
                        Some(current_sum) => {
                            let tmp_var = variables_counter;
                            variables_counter += 1;
                            circuits.push(Circuit::Add(tmp_var, current_sum, val));
                            Some(CircuitValue::Variable(tmp_var))
                        }
                        None => Some(val),
                    };
                }
                (sum_var.unwrap(), variables_counter, circuits)
            }

            Expression::Product(exprs) => {
                let mut prod_var = None;
                let mut circuits = vec![];

                for expr in exprs {
                    let (val, next_counter, new_circuits) = flatten_expr(variables_counter, expr);
                    variables_counter = next_counter;
                    circuits.extend(new_circuits);

                    prod_var = match prod_var {
                        Some(current_prod) => {
                            let tmp_var = variables_counter;
                            variables_counter += 1;
                            circuits.push(Circuit::Mult(tmp_var, current_prod, val));
                            Some(CircuitValue::Variable(tmp_var))
                        }
                        None => Some(val),
                    };
                }
                (prod_var.unwrap(), variables_counter, circuits)
            }
        }
    }

    let (_, id, circuits) = flatten_expr(variables_counter, expression);
    (id, circuits)
}

pub fn remove_output(exprs: &[Expression]) -> Vec<Expression> {
    exprs
        .par_iter()
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

pub fn replace_input_expr(expr: Expression, from: Id, to: Id) -> Expression {
    match expr {
        Expression::Input(id) if id == from => Expression::Input(to),
        Expression::Eq(lhs, rhs) => Expression::Eq(
            Box::new(replace_input_expr(*lhs, from, to)),
            Box::new(replace_input_expr(*rhs, from, to)),
        ),
        Expression::Sum(exprs) => Expression::Sum(
            exprs
                .into_iter()
                .map(|expr| replace_input_expr(expr, from, to))
                .collect(),
        ),
        Expression::Product(exprs) => Expression::Product(
            exprs
                .into_iter()
                .map(|expr| replace_input_expr(expr, from, to))
                .collect(),
        ),
        expr => expr,
    }
}

pub fn replace_var(exprs: &Vec<Expression>, from: Id, to: Id) -> Vec<Expression> {
    exprs
        .par_iter()
        .map(|expr| replace_var_expr(expr.clone(), from, to))
        .collect()
}

pub fn replace_input(exprs: &Vec<Expression>, from: Id, to: Id) -> Vec<Expression> {
    exprs
        .par_iter()
        .map(|expr| replace_input_expr(expr.clone(), from, to))
        .collect()
}

pub fn replace_input2var(expr: Expression, from: Id, to: Id) -> Expression {
    match expr {
        Expression::Input(id) if id == from => Expression::Variable(to),
        Expression::Eq(lhs, rhs) => Expression::Eq(
            Box::new(replace_input2var(*lhs, from, to)),
            Box::new(replace_input2var(*rhs, from, to)),
        ),
        Expression::Sum(exprs) => Expression::Sum(
            exprs
                .into_iter()
                .map(|expr| replace_input2var(expr, from, to))
                .collect(),
        ),
        Expression::Product(exprs) => Expression::Product(
            exprs
                .into_iter()
                .map(|expr| replace_input2var(expr, from, to))
                .collect(),
        ),
        expr => expr,
    }
}

pub fn find_max_var_id(exprs: &[Expression]) -> Id {
    exprs
        .par_iter()
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
            Expression::Sum(exprs) | Expression::Product(exprs) => vec![find_max_var_id(exprs)],
            Expression::Input(_) | Expression::Constant(_) => vec![],
        })
        .max()
        .unwrap_or(0)
}

pub fn find_max_input_id(exprs: &[Expression]) -> Id {
    exprs
        .par_iter()
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

pub fn concat_exprs(conv1: Vec<Expression>, mut conv2: Vec<Expression>) -> Vec<Expression> {
    if conv1.is_empty() {
        return conv2;
    }

    let mut inputs1 = vec![];
    for expr in &conv1 {
        if let Expression::Eq(var, input) = expr {
            if let (&Expression::Variable(_), &Expression::Input(id)) =
                (var.as_ref(), input.as_ref())
            {
                inputs1.push(id);
            }
        }
    }

    let mut inputs2 = vec![];
    for expr in &conv2 {
        if let Expression::Eq(var, input) = expr {
            if let (&Expression::Variable(_), &Expression::Input(id)) =
                (var.as_ref(), input.as_ref())
            {
                inputs2.push(id);
            }
        }
    }

    let exprs2 = remove_output(&conv2);
    let max_id = find_max_var_id(&conv1);
    let exprs1 = remove_output(&conv1);
    for (old_id, new_id) in (0..=max_id).zip(max_id + 1..) {
        conv2 = replace_var(&conv2, old_id, new_id);
    }

    for (from, to) in ((max_id - inputs1.len() + 1)..=max_id).enumerate() {
        conv2 = conv2
            .into_iter()
            .map(|expr| replace_input2var(expr, from, to))
            .collect();
    }

    // 入力サイズが異なる場合の調整
    let diff = max_input_id(&exprs1) - max_input_id(&exprs2);
    for input2 in inputs2.into_iter() {
        conv2 = replace_input(&conv2, input2, input2 + diff);
    }

    exprs1.into_iter().chain(conv2).collect()
}

pub fn multiple_layer(exprs1: Vec<Expression>, exprs2: Vec<Expression>) -> Vec<Expression> {
    let mut inputs1 = vec![];
    for expr in &exprs1 {
        if let Expression::Eq(var, input) = expr {
            if let (&Expression::Variable(_), &Expression::Input(_)) =
                (var.as_ref(), input.as_ref())
            {
                inputs1.push(input.clone());
            }
        }
    }
    let mut inputs2 = vec![];
    for expr in &exprs1 {
        if let Expression::Eq(var, input) = expr {
            if let (&Expression::Variable(_), &Expression::Input(_)) =
                (var.as_ref(), input.as_ref())
            {
                inputs2.push(input.clone());
            }
        }
    }
    assert_eq!(inputs1, inputs2);

    let max_id = find_max_var_id(&exprs1);
    let mut exprs = remove_output(&exprs1);
    let mut exprs2 = remove_output(&exprs2);
    for (old_id, new_id) in (0..=max_id).zip(max_id + 1..) {
        exprs2 = replace_var(&exprs2, old_id, new_id);
    }
    exprs.extend(exprs2);
    for i in 0..inputs1.len() {
        exprs.push(Expression::Eq(
            Box::new(Expression::Variable(i + (max_id + 1) * 2)),
            Box::new(Expression::Product(vec![
                Expression::Variable(i),
                Expression::Variable(i + max_id + 1),
            ])),
        ));
    }

    for (i, input) in inputs1.into_iter().enumerate() {
        exprs.push(Expression::Eq(
            Box::new(Expression::Variable(i + (max_id + 1) * 2)),
            input,
        ));
    }

    exprs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cons(value: i64) -> Expression {
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
        let model = Dense {
            weight: vec![vec![1, 2]],
            bias: vec![3],
        };
        assert_eq!(
            linear_to_exprs(model),
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

        let model = Dense {
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        assert_eq!(
            linear_to_exprs(model),
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

        let model = Dense {
            weight: vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![9, 10, 11, 12],
                vec![13, 14, 15, 16],
            ],
            bias: vec![17, 18, 19, 20],
        };

        assert_eq!(
            linear_to_exprs(model),
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

    fn ccons(value: i64) -> CircuitValue {
        CircuitValue::Constant(value)
    }

    #[test]
    fn test_flatten() {
        // v0 = i0 * 1 + i1 * 2 + 3
        let expr = eq(
            var(0),
            sum(&[
                prod(&[input(0), cons(1)]),
                prod(&[input(1), cons(2)]),
                cons(3),
            ]),
        );
        let var_counter = 10;
        assert_eq!(
            flatten(expr, var_counter).1,
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
        let model = Dense {
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };

        let exprs = linear_to_exprs(model);
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
            replace_var(&vec![expr.clone()], 0, 3),
            vec![eq(var(3), sum(&[var(1), var(2)]),)]
        );
        assert_eq!(
            replace_var(&vec![expr.clone()], 1, 3),
            vec![eq(var(0), sum(&[var(3), var(2)]),)]
        );
        assert_eq!(
            replace_var(&vec![expr.clone()], 2, 3),
            vec![eq(var(0), sum(&[var(1), var(3)]),)]
        );
    }

    #[test]
    fn test_multi_layer() {
        let layer1 = Dense {
            weight: vec![vec![1]],
            bias: vec![1],
        };
        let layer2 = Dense {
            weight: vec![vec![1]],
            bias: vec![1],
        };

        let exprs1 = linear_to_exprs(layer1);
        let exprs2 = linear_to_exprs(layer2);
        assert_eq!(
            concat_exprs(exprs1, exprs2),
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(1)])),
                eq(var(1), sum(&[prod(&[var(0), cons(1)]), cons(1)])),
                eq(var(1), input(1))
            ],
        );

        let layer1 = Dense {
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let layer2 = Dense {
            weight: vec![vec![1, 2], vec![3, 4]],
            bias: vec![5, 6],
        };
        let exprs1 = linear_to_exprs(layer1);
        let exprs2 = linear_to_exprs(layer2);
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

        let conv1 = Conv {
            input_size: 1,
            stride: 1,
            weight: vec![vec![vec![vec![1]]]],
            bias: vec![1],
        };
        let conv2 = Conv {
            input_size: 1,
            stride: 1,
            weight: vec![vec![vec![vec![1]]]],
            bias: vec![1],
        };

        let exprs1 = conv_to_exprs(conv1);
        assert_eq!(
            exprs1,
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(1)])),
                eq(var(0), input(1)),
            ]
        );
        let exprs2 = conv_to_exprs(conv2);
        assert_eq!(
            exprs2,
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(1)])),
                eq(var(0), input(1)),
            ]
        );

        let exprs = concat_exprs(exprs1, exprs2);
        assert_eq!(
            exprs,
            vec![
                eq(var(0), sum(&[prod(&[input(0), cons(1)]), cons(1)])),
                eq(var(1), sum(&[prod(&[var(0), cons(1)]), cons(1)])),
                eq(var(1), input(1))
            ]
        );

        assert_eq!(get_variables(&exprs, vec![1, 3]), vec![2, 3]);
    }

    #[test]
    fn test_gen_conv_input() {
        assert_eq!(gen_conv_input(0, 2, 3), vec![vec![0, 1], vec![3, 4]]);
        assert_eq!(gen_conv_input(1, 2, 3), vec![vec![1, 2], vec![4, 5]]);
        assert_eq!(gen_conv_input(3, 2, 3), vec![vec![3, 4], vec![6, 7]]);
        assert_eq!(gen_conv_input(4, 2, 3), vec![vec![4, 5], vec![7, 8]]);
    }

    #[test]
    fn test_conv() {
        let conv = Conv {
            input_size: 1,
            stride: 1,
            weight: vec![vec![vec![vec![12]]]],
            bias: vec![7],
        };

        let exprs = conv_to_exprs(conv);

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

        let exprs = conv_to_exprs(conv);

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
        let exprs = conv_to_exprs(conv);

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
        let exprs = conv_to_exprs(conv);

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
        let exprs = conv_to_exprs(conv);

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
    fn test_conv_multi_input_channel() {
        let conv = Conv {
            input_size: 2,
            stride: 1,
            weight: vec![vec![
                vec![vec![3, 3], vec![3, 3]],
                vec![vec![3, 3], vec![3, 3]],
            ]],
            bias: vec![5],
        };
        let exprs = conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(3)]),
                        prod(&[input(2), cons(3)]),
                        prod(&[input(3), cons(3)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(5), cons(3)]),
                        prod(&[input(6), cons(3)]),
                        prod(&[input(7), cons(3)]),
                        cons(5)
                    ])
                ),
                eq(var(0), input(8)),
            ]
        );
    }

    #[test]
    fn test_conv_multi_output_channel() {
        let conv = Conv {
            input_size: 3,
            stride: 1,
            weight: vec![
                vec![vec![vec![3, 3], vec![3, 3]], vec![vec![3, 3], vec![3, 3]]],
                vec![vec![vec![4, 4], vec![4, 4]], vec![vec![4, 4], vec![4, 4]]],
            ],
            bias: vec![5, 8],
        };
        let exprs = conv_to_exprs(conv);

        assert_eq!(
            exprs,
            vec![
                // output 0
                eq(
                    var(0),
                    sum(&[
                        prod(&[input(0), cons(3)]),
                        prod(&[input(1), cons(3)]),
                        prod(&[input(3), cons(3)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(9), cons(3)]),
                        prod(&[input(10), cons(3)]),
                        prod(&[input(12), cons(3)]),
                        prod(&[input(13), cons(3)]),
                        cons(5)
                    ])
                ),
                eq(
                    var(1),
                    sum(&[
                        prod(&[input(1), cons(3)]),
                        prod(&[input(2), cons(3)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(5), cons(3)]),
                        prod(&[input(10), cons(3)]),
                        prod(&[input(11), cons(3)]),
                        prod(&[input(13), cons(3)]),
                        prod(&[input(14), cons(3)]),
                        cons(5)
                    ])
                ),
                eq(
                    var(2),
                    sum(&[
                        prod(&[input(3), cons(3)]),
                        prod(&[input(4), cons(3)]),
                        prod(&[input(6), cons(3)]),
                        prod(&[input(7), cons(3)]),
                        prod(&[input(12), cons(3)]),
                        prod(&[input(13), cons(3)]),
                        prod(&[input(15), cons(3)]),
                        prod(&[input(16), cons(3)]),
                        cons(5)
                    ])
                ),
                eq(
                    var(3),
                    sum(&[
                        prod(&[input(4), cons(3)]),
                        prod(&[input(5), cons(3)]),
                        prod(&[input(7), cons(3)]),
                        prod(&[input(8), cons(3)]),
                        prod(&[input(13), cons(3)]),
                        prod(&[input(14), cons(3)]),
                        prod(&[input(16), cons(3)]),
                        prod(&[input(17), cons(3)]),
                        cons(5)
                    ])
                ),
                // output 1
                eq(
                    var(4),
                    sum(&[
                        prod(&[input(0), cons(4)]),
                        prod(&[input(1), cons(4)]),
                        prod(&[input(3), cons(4)]),
                        prod(&[input(4), cons(4)]),
                        prod(&[input(9), cons(4)]),
                        prod(&[input(10), cons(4)]),
                        prod(&[input(12), cons(4)]),
                        prod(&[input(13), cons(4)]),
                        cons(8)
                    ])
                ),
                eq(
                    var(5),
                    sum(&[
                        prod(&[input(1), cons(4)]),
                        prod(&[input(2), cons(4)]),
                        prod(&[input(4), cons(4)]),
                        prod(&[input(5), cons(4)]),
                        prod(&[input(10), cons(4)]),
                        prod(&[input(11), cons(4)]),
                        prod(&[input(13), cons(4)]),
                        prod(&[input(14), cons(4)]),
                        cons(8)
                    ])
                ),
                eq(
                    var(6),
                    sum(&[
                        prod(&[input(3), cons(4)]),
                        prod(&[input(4), cons(4)]),
                        prod(&[input(6), cons(4)]),
                        prod(&[input(7), cons(4)]),
                        prod(&[input(12), cons(4)]),
                        prod(&[input(13), cons(4)]),
                        prod(&[input(15), cons(4)]),
                        prod(&[input(16), cons(4)]),
                        cons(8)
                    ])
                ),
                eq(
                    var(7),
                    sum(&[
                        prod(&[input(4), cons(4)]),
                        prod(&[input(5), cons(4)]),
                        prod(&[input(7), cons(4)]),
                        prod(&[input(8), cons(4)]),
                        prod(&[input(13), cons(4)]),
                        prod(&[input(14), cons(4)]),
                        prod(&[input(16), cons(4)]),
                        prod(&[input(17), cons(4)]),
                        cons(8)
                    ])
                ),
                eq(var(0), input(18)),
                eq(var(1), input(19)),
                eq(var(2), input(20)),
                eq(var(3), input(21)),
                eq(var(4), input(22)),
                eq(var(5), input(23)),
                eq(var(6), input(24)),
                eq(var(7), input(25)),
            ]
        );
    }

    #[test]
    fn test_multiple_layer() {
        let exprs1 = vec![
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
            ),
            eq(var(0), input(16)),
            eq(var(1), input(17)),
            eq(var(2), input(18)),
            eq(var(3), input(19)),
        ];

        let exprs2 = vec![
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
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
                    cons(10),
                ]),
            ),
            eq(var(0), input(16)),
            eq(var(1), input(17)),
            eq(var(2), input(18)),
            eq(var(3), input(19)),
        ];

        assert_eq!(
            multiple_layer(exprs1, exprs2),
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
                        cons(10),
                    ]),
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
                        cons(10),
                    ]),
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
                        cons(10),
                    ]),
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
                        cons(10),
                    ]),
                ),
                eq(
                    var(4),
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
                        cons(10),
                    ]),
                ),
                eq(
                    var(5),
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
                        cons(10),
                    ]),
                ),
                eq(
                    var(6),
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
                        cons(10),
                    ]),
                ),
                eq(
                    var(7),
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
                        cons(10),
                    ]),
                ),
                eq(var(8), prod(&[var(0), var(4)])),
                eq(var(9), prod(&[var(1), var(5)])),
                eq(var(10), prod(&[var(2), var(6)])),
                eq(var(11), prod(&[var(3), var(7)])),
                eq(var(8), input(16)),
                eq(var(9), input(17)),
                eq(var(10), input(18)),
                eq(var(11), input(19)),
            ]
        );
    }
}
