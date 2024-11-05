use crate::{
    circuit::Circuit,
    compiler::{concat_exprs, find_max_id, find_max_input_id, Compiler},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<u32>,
    pub bias: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    layers: Vec<Linear>,
}

impl Model {
    pub fn new(linear: Vec<Linear>) -> Self {
        Self { layers: linear }
    }
}

impl Model {
    pub fn circuits(&self) -> Vec<Circuit> {
        let mut exprs = vec![];
        for layer in self.layers.iter() {
            let mut compiler = Compiler::new();
            let new_exprs = compiler.linear_to_expression(layer.clone());
            exprs = concat_exprs(exprs, new_exprs);
        }
        let max_var_id = find_max_id(&exprs);
        let max_input_id = find_max_input_id(&exprs);
        let mut compiler = Compiler::new_with_counters(max_input_id, max_var_id + 1);
        exprs
            .into_iter()
            .flat_map(|expr| compiler.flatten(expr))
            .collect()
    }

    pub fn compute(&self, input: &[u32]) -> Vec<u32> {
        let mut output = input.to_vec();
        for layer in self.layers.iter() {
            let mut new_output = vec![0u32; layer.output];
            (0..layer.output).for_each(|i| {
                new_output[i] = layer.bias[i];
                (0..layer.input).for_each(|j| {
                    new_output[i] += layer.weight[i * layer.input + j] * output[j];
                });
            });
            output = new_output;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::CircuitValue;

    use super::*;

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
    fn test_circits() {
        let model = Model {
            layers: vec![Linear {
                input: 2,
                output: 1,
                weight: vec![1, 2],
                bias: vec![3],
            }],
        };
        // to expressions
        // v0 = i0 * 1 + i1 * 2 + 3
        // v0 = i2
        //
        // to circuits
        // v1 = i0 * 1
        // v2 = i1 * 2
        // v3 = v1 + v2
        // v4 = v3 + 3
        // v0 = v4
        // v0 = i2

        let circuits = model.circuits();
        assert_eq!(
            circuits,
            // v1 = i0 * 1
            // v2 = i1 * 2
            // v3 = v1 + v2
            // v4 = v3 + 3
            // v0 = v4
            // v0 = i2
            vec![
                cmult(1, cinput(0), ccons(1)),
                cmult(2, cinput(1), ccons(2)),
                cadd(3, cvar(1), cvar(2)),
                cadd(4, cvar(3), ccons(3)),
                ceq(0, cvar(4)),
                ceq(0, cinput(2)),
            ]
        );
    }

    #[test]
    fn test_multi_layer() {
        let model = Model::new(vec![
            Linear {
                input: 1,
                output: 1,
                weight: vec![1],
                bias: vec![1],
            },
            Linear {
                input: 1,
                output: 1,
                weight: vec![1],
                bias: vec![1],
            },
        ]);
        let circuits = model.circuits();
        // [Eq(Variable(0), Sum([Product([Input(0), Constant(1)]), Constant(1)]))
        //  Eq(Variable(1), Sum([Product([Variable(0), Constant(1)]), Constant(1)]))
        //  Eq(Variable(1), Input(1))]

        // [
        // Eq(Variable(0), Sum([Product([Input(0), Constant(1)]), Constant(1)]))
        // Eq(Variable(1), Sum([Product([Input(0), Constant(1)]), Constant(1)]))
        // Eq(Variable(1), Input(1))
        // ]
        // v0 = i0 * 1 + 1
        // v1 = i0 * 1 + 1
        // v1 = i1

        // [
        // Mult(2, Input(0), Constant(1)),
        // Add(3, Variable(2), Constant(1)),
        // Eq(0, Variable(3)),
        // Mult(4, Variable(0), Constant(1)),
        // Add(5, Variable(4), Constant(1)),
        // Eq(1, Variable(5)),
        // Eq(1, Input(1))]
        //
        // Mult(2, Input(0), Constant(1)),
        // Add(3, Variable(2), Constant(1)),
        // Mult(4, Variable(3), Constant(1)),
        // Add(5, Variable(4),Constant(1)),
        // Eq(1, Variable(5))]
        assert_eq!(
            circuits,
            vec![
                cmult(2, cinput(0), ccons(1)),
                cadd(3, cvar(2), ccons(1)),
                cmult(4, cvar(3), ccons(1)),
                cadd(5, cvar(4), ccons(1)),
                ceq(1, cvar(5)),
            ]
        );
    }
}
