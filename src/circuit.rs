use crate::compiler::Id;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Circuit {
    // x = a
    Eq(Id, CircuitValue),
    // x = a + b
    Mult(Id, CircuitValue, CircuitValue),
    // x = a * b
    Add(Id, CircuitValue, CircuitValue),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitValue {
    Variable(Id),
    Input(Id),
    Constant(u32),
}

pub fn get_variables(circuits: &[Circuit], input: Vec<u32>) -> Vec<u32> {
    let mut vars = vec![0u32; num_vars(circuits)];

    for circuit in circuits {
        match *circuit {
            Circuit::Eq(id, value) => {
                let computed_value = evaluate_value(&value, &vars, &input);
                vars[id] = computed_value;
            }
            Circuit::Add(id, val1, val2) => {
                let computed_val1 = evaluate_value(&val1, &vars, &input);
                let computed_val2 = evaluate_value(&val2, &vars, &input);
                vars[id] = computed_val1 + computed_val2;
            }
            Circuit::Mult(id, val1, val2) => {
                let computed_val1 = evaluate_value(&val1, &vars, &input);
                let computed_val2 = evaluate_value(&val2, &vars, &input);
                vars[id] = computed_val1 * computed_val2;
            }
        }
    }

    vars
}

fn evaluate_value(value: &CircuitValue, vars: &[u32], input: &[u32]) -> u32 {
    match *value {
        CircuitValue::Variable(id) => vars[id],
        CircuitValue::Input(id) => input[id],
        CircuitValue::Constant(val) => val,
    }
}

pub fn num_vars(circuits: &[Circuit]) -> usize {
    circuits
        .iter()
        .flat_map(|c| match c {
            Circuit::Eq(id1, CircuitValue::Variable(id2)) => vec![*id1, *id2],
            Circuit::Add(id1, _, CircuitValue::Variable(id2)) => vec![*id1, *id2],
            Circuit::Add(id1, CircuitValue::Variable(id2), _) => vec![*id1, *id2],
            Circuit::Mult(id1, _, CircuitValue::Variable(id2)) => vec![*id1, *id2],
            Circuit::Mult(id1, CircuitValue::Variable(id2), _) => vec![*id1, *id2],
            Circuit::Eq(id, _) | Circuit::Add(id, _, _) | Circuit::Mult(id, _, _) => vec![*id],
        })
        .max()
        .map(|max_id| max_id + 1)
        .unwrap_or(0)
}

pub fn num_inputs(circuits: &[Circuit]) -> usize {
    circuits
        .iter()
        .filter_map(|c| match c {
            Circuit::Eq(_, CircuitValue::Input(id))
            | Circuit::Add(_, CircuitValue::Input(id), _)
            | Circuit::Mult(_, CircuitValue::Input(id), _)
            | Circuit::Add(_, _, CircuitValue::Input(id))
            | Circuit::Mult(_, _, CircuitValue::Input(id)) => Some(*id),
            _ => None,
        })
        .max()
        .map(|max_id| max_id + 1)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_vars_no_duplicates() {
        let circuits = vec![
            Circuit::Eq(0, CircuitValue::Variable(1)),
            Circuit::Add(2, CircuitValue::Variable(1), CircuitValue::Input(0)),
            Circuit::Mult(3, CircuitValue::Variable(2), CircuitValue::Constant(10)),
        ];
        assert_eq!(num_vars(&circuits), 4);
    }

    #[test]
    fn test_num_vars_with_duplicates() {
        let circuits = vec![
            Circuit::Eq(0, CircuitValue::Variable(1)),
            Circuit::Add(1, CircuitValue::Variable(0), CircuitValue::Input(0)),
            Circuit::Mult(0, CircuitValue::Variable(1), CircuitValue::Constant(10)),
        ];
        assert_eq!(num_vars(&circuits), 2);
    }

    #[test]
    fn test_num_vars_with_inputs() {
        let circuits = vec![
            Circuit::Eq(0, CircuitValue::Input(1)),
            Circuit::Add(1, CircuitValue::Input(0), CircuitValue::Constant(5)),
        ];
        assert_eq!(num_vars(&circuits), 2);
    }

    #[test]
    fn test_num_vars_only_constants() {
        let circuits = vec![
            Circuit::Eq(0, CircuitValue::Constant(1)),
            Circuit::Add(1, CircuitValue::Constant(2), CircuitValue::Constant(3)),
        ];
        assert_eq!(num_vars(&circuits), 2);
    }

    #[test]
    fn test_num_inputs() {
        let circuits = vec![
            Circuit::Eq(0, CircuitValue::Input(1)),
            Circuit::Add(1, CircuitValue::Input(0), CircuitValue::Constant(5)),
        ];
        assert_eq!(num_inputs(&circuits), 2);
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
    fn test_get_variables() {
        // v0 = i0 * 1
        // v1 = i1 * 2
        // v2 = v0 + v1
        // v3 = v2 + 3
        // v4 = v3 * v1
        // v4 = i2
        let circuits = vec![
            cmult(0, cinput(0), ccons(1)),
            cmult(1, cinput(1), ccons(2)),
            cadd(2, cvar(0), cvar(1)),
            cadd(3, cvar(2), ccons(3)),
            cmult(4, cvar(3), cvar(1)),
            ceq(4, cinput(2)),
        ];
        let input = vec![2, 3, 66];
        assert_eq!(get_variables(&circuits, input), vec![2, 6, 8, 11, 66]);
    }
}
