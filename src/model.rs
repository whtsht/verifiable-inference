#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<u32>,
    pub bias: Vec<u32>,
}
