use crate::{circuit::Circuit, compiler::Compiler};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<u32>,
    pub bias: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    linear: Linear,
}

impl Model {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
}

impl Model {
    pub fn circuits(&self) -> Vec<Circuit> {
        let mut compiler = Compiler::new();
        let exprs = compiler.linear_to_expression(self.linear.clone());
        exprs
            .into_iter()
            .flat_map(|expr| compiler.flatten(expr))
            .collect()
    }

    pub fn compute(&self, input: &[u32]) -> Vec<u32> {
        let mut output = vec![0u32; self.linear.output];

        for i in 0..self.linear.output {
            output[i] = self.linear.bias[i];
            (0..self.linear.input).for_each(|j| {
                output[i] += self.linear.weight[i * self.linear.input + j] * input[j];
            });
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
            linear: Linear {
                input: 2,
                output: 1,
                weight: vec![1, 2],
                bias: vec![3],
            },
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
}
