use crate::{
    circuit::Circuit,
    compiler::{
        concat_exprs, conv_to_exprs, find_max_var_id, flatten, linear_to_exprs, multiple_layer,
        Expression,
    },
    jq::fetch_value,
};

pub trait Layer {
    type Input;
    type Output;
    fn forward(&self, input: Self::Input) -> Self::Output;
}

#[derive(Debug, Clone)]
pub struct Dense {
    pub weight: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

impl Layer for Dense {
    type Input = Vec<i64>;

    type Output = Vec<i64>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        self.weight
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let sum: i64 = row.iter().zip(&input).map(|(w, &x)| w * x).sum();
                sum + self.bias[i]
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Conv {
    //pub kernel_size: usize,
    pub stride: usize,
    //pub channel_size: usize,
    // weight: [out_channel][in_channel][kernel_size][kernel_size]
    pub weight: Vec<Vec<Vec<Vec<i64>>>>,
    // bias: [out_channel]
    pub bias: Vec<i64>,

    pub input_size: usize,
}

impl Layer for Conv {
    // input, output: [channel_size][height][width]
    type Input = Vec<Vec<Vec<i64>>>;
    type Output = Vec<Vec<Vec<i64>>>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let input_size = input[0].len();
        let kernel_size = self.weight[0][0].len();
        let channel_size = self.weight.len();
        let output_size = (input_size - kernel_size) / self.stride + 1;

        let mut output = vec![vec![vec![0; output_size]; output_size]; channel_size];

        for (out_channel_idx, out_channel) in output.iter_mut().enumerate() {
            for (in_channel_idx, in_channel) in input.iter().enumerate() {
                for y_start in (0..output_size).map(|y| y * self.stride) {
                    for x_start in (0..output_size).map(|x| x * self.stride) {
                        let mut sum = 0;

                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let y = y_start + ky;
                                let x = x_start + kx;

                                sum += in_channel[y][x]
                                    * self.weight[out_channel_idx][in_channel_idx][ky][kx];
                            }
                        }

                        let out_y = y_start / self.stride;
                        let out_x = x_start / self.stride;
                        out_channel[out_y][out_x] += sum;
                    }
                }
            }

            for row in out_channel.iter_mut() {
                for value in row.iter_mut() {
                    *value += self.bias[out_channel_idx];
                }
            }
        }

        output
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub conv11: Conv,
    pub conv12: Conv,
    pub conv21: Conv,
    pub conv22: Conv,
    pub fc: Dense,
}

const SCALE: f64 = 100.0;

pub fn fetch_bias(filename: &str, layer: &str) -> Vec<i64> {
    let bias: Vec<f64> = fetch_value(filename, &format!("{}.bias", layer)).unwrap();
    let max_bias = bias.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let scale = SCALE / max_bias;
    bias.into_iter()
        .map(|x| (x * scale).round() as i64)
        .collect()
}

pub fn fetch_conv_weight(filename: &str, layer: &str) -> Vec<Vec<Vec<Vec<i64>>>> {
    let weight: Vec<Vec<Vec<Vec<f64>>>> =
        fetch_value(filename, &format!("{}.weight", layer)).unwrap();

    let max_weight = weight
        .iter()
        .flat_map(|x| x.iter().flat_map(|y| y.iter().flat_map(|z| z.iter())))
        .fold(0.0f64, |a, &b| a.max(b.abs()));

    let scale = SCALE / max_weight;

    weight
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(|y| {
                    y.into_iter()
                        .map(|z| z.into_iter().map(|a| (a * scale).round() as i64).collect())
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn fetch_dense_weight(filename: &str, layer: &str) -> Vec<Vec<i64>> {
    let weight: Vec<Vec<f64>> = fetch_value(filename, &format!("{}.weight", layer)).unwrap();
    let max_weight = weight
        .iter()
        .flat_map(|x| x.iter())
        .fold(0.0f64, |a, &b| a.max(b.abs()));

    let scale = SCALE / max_weight;

    weight
        .into_iter()
        .map(|x| x.into_iter().map(|y| (y * scale).round() as i64).collect())
        .collect()
}

impl Model {
    pub fn load() -> Self {
        let filename = "pi_net.json";
        let conv11 = Conv {
            input_size: 28,
            stride: 1,
            bias: fetch_bias(filename, "conv11"),
            weight: fetch_conv_weight(filename, "conv11"),
        };

        let conv12 = Conv {
            input_size: 28,
            stride: 1,
            bias: fetch_bias(filename, "conv12"),
            weight: fetch_conv_weight(filename, "conv12"),
        };

        let conv21 = Conv {
            input_size: 26,
            stride: 2,
            bias: fetch_bias(filename, "conv21"),
            weight: fetch_conv_weight(filename, "conv21"),
        };

        let conv22 = Conv {
            input_size: 26,
            stride: 2,
            bias: fetch_bias(filename, "conv22"),
            weight: fetch_conv_weight(filename, "conv22"),
        };

        let fc = Dense {
            bias: fetch_bias(filename, "fc"),
            weight: fetch_dense_weight(filename, "fc"),
        };

        Model {
            conv11,
            conv12,
            conv21,
            conv22,
            fc,
        }
    }
}

impl Model {
    pub fn compute(&self, input: &[u8]) -> Vec<i64> {
        let input: Vec<Vec<i64>> = input
            .chunks(28)
            .map(|row| row.iter().map(|&x| x as i64).collect())
            .collect();

        let conv11_out = self.conv11.forward(vec![input.clone()]);
        let conv12_out = self.conv12.forward(vec![input]);
        //assert_eq!(conv11_out.len(), 6);
        //assert_eq!(conv11_out[0].len(), 26);
        //assert_eq!(conv11_out[0][0].len(), 26);
        //assert_eq!(conv12_out.len(), 6);
        //assert_eq!(conv12_out[0].len(), 26);
        //assert_eq!(conv12_out[0][0].len(), 26);
        let conv1_img_len = conv11_out[0].len();
        let conv1_out_len = conv11_out.len();
        let mut conv1_out = vec![vec![vec![0; conv1_img_len]; conv1_img_len]; conv1_out_len];
        for i in 0..conv1_out_len {
            for j in 0..conv1_img_len {
                for k in 0..conv1_img_len {
                    conv1_out[i][j][k] = conv11_out[i][j][k] * conv12_out[i][j][k];
                }
            }
        }

        let conv21_out = self.conv21.forward(conv1_out.clone());
        let conv22_out = self.conv22.forward(conv1_out);
        //assert_eq!(conv21_out.len(), 12);
        //assert_eq!(conv21_out[0].len(), 12);
        //assert_eq!(conv21_out[0][0].len(), 12);
        //assert_eq!(conv22_out.len(), 12);
        //assert_eq!(conv22_out[0].len(), 12);
        //assert_eq!(conv22_out[0][0].len(), 12);
        let conv2_img_len = conv21_out[0].len();
        let conv2_out_len = conv21_out.len();
        let mut conv2_out = vec![vec![vec![0; conv2_img_len]; conv2_img_len]; conv2_out_len];
        for i in 0..conv2_out_len {
            for j in 0..conv2_img_len {
                for k in 0..conv2_img_len {
                    conv2_out[i][j][k] = conv21_out[i][j][k] * conv22_out[i][j][k];
                }
            }
        }

        let fc_input: Vec<i64> = conv2_out
            .into_iter()
            .flat_map(|channel| channel.into_iter().flat_map(|row| row.into_iter()))
            .collect();

        self.fc.forward(fc_input)
    }

    fn exprs(&self) -> Vec<Expression> {
        let model = self.clone();
        let conv11 = conv_to_exprs(model.conv11);
        let conv12 = conv_to_exprs(model.conv12);
        let conv21 = conv_to_exprs(model.conv21);
        let conv22 = conv_to_exprs(model.conv22);
        let fc = linear_to_exprs(model.fc);
        let conv1 = multiple_layer(conv11, conv12);
        let conv2 = multiple_layer(conv21, conv22);
        let mut exprs = vec![];
        exprs = concat_exprs(exprs, conv1);
        exprs = concat_exprs(exprs, conv2);
        exprs = concat_exprs(exprs, fc);
        exprs
    }

    pub fn circuits(&self) -> Vec<Circuit> {
        let exprs = self.exprs();
        let mut variable_counter = find_max_var_id(&exprs) + 1;
        let mut circuits = vec![];
        for expr in exprs {
            let (next_var, circuit) = flatten(expr, variable_counter);
            circuits.extend(circuit);
            variable_counter = next_var;
        }

        circuits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::concat_exprs;

    #[test]
    fn test_circuits() {
        let model = Model {
            conv11: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 4,
            },
            conv12: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 4,
            },
            conv21: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 3,
            },
            conv22: Conv {
                stride: 1,
                weight: vec![vec![vec![vec![1, 1], vec![1, 1]]]],
                bias: vec![1],
                input_size: 3,
            },
            fc: Dense {
                weight: vec![
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                ],
                bias: vec![1, 1, 1, 1],
            },
        };

        let conv11 = conv_to_exprs(model.conv11);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        ];
        assert_eq!(
            crate::compiler::get_variables(&conv11, input),
            vec![5, 5, 5, 5, 5, 5, 5, 5, 5]
        );
        let conv12 = conv_to_exprs(model.conv12);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        ];
        assert_eq!(
            crate::compiler::get_variables(&conv12, input),
            vec![5, 5, 5, 5, 5, 5, 5, 5, 5]
        );
        let conv21 = conv_to_exprs(model.conv21);
        let input = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 21, 21, 21, 21];
        assert_eq!(
            crate::compiler::get_variables(&conv21, input),
            vec![21, 21, 21, 21]
        );
        let conv22 = conv_to_exprs(model.conv22);
        let input = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 21, 21, 21, 21];
        assert_eq!(
            crate::compiler::get_variables(&conv22, input),
            vec![21, 21, 21, 21]
        );
        let fc = linear_to_exprs(model.fc);
        let input = vec![21, 21, 21, 21, 85, 85, 85, 85];
        assert_eq!(
            crate::compiler::get_variables(&fc, input),
            vec![85, 85, 85, 85]
        );

        let conv1 = multiple_layer(conv11, conv12);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        ];
        assert_eq!(
            crate::compiler::get_variables(&conv1, input),
            vec![
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 25, 25, 25, 25, 25, 25, 25,
                25, 25
            ]
        );
        let conv2 = multiple_layer(conv21, conv22);
        let input = vec![
            25, 25, 25, 25, 25, 25, 25, 25, 25, 10201, 10201, 10201, 10201,
        ];
        assert_eq!(
            crate::compiler::get_variables(&conv2, input),
            vec![101, 101, 101, 101, 101, 101, 101, 101, 10201, 10201, 10201, 10201]
        );

        let mut exprs = vec![];
        exprs = concat_exprs(exprs, conv1);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // input
            25, 25, 25, 25, 25, 25, 25, 25, 25, // output
        ];
        assert_eq!(
            crate::compiler::get_variables(&exprs, input),
            vec![
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 25, 25, 25, 25, 25, 25, 25,
                25, 25
            ]
        );
        exprs = concat_exprs(exprs, conv2);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10201, 10201, 10201, 10201,
        ];
        assert_eq!(
            crate::compiler::get_variables(&exprs, input),
            vec![
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, // conv11, conv12
                25, 25, 25, 25, 25, 25, 25, 25, 25, // conv11 * conv12
                101, 101, 101, 101, 101, 101, 101, 101, // conv21, conv22
                10201, 10201, 10201, 10201 // conv21 * conv22
            ]
        );
        exprs = concat_exprs(exprs, fc);
        let input = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 40805, 40805, 40805, 40805,
        ];
        assert_eq!(
            crate::compiler::get_variables(&exprs, input),
            vec![
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, // conv11, conv12
                25, 25, 25, 25, 25, 25, 25, 25, 25, // conv11 * conv12
                101, 101, 101, 101, 101, 101, 101, 101, // conv21, conv22
                10201, 10201, 10201, 10201, // conv21 * conv22
                40805, 40805, 40805, 40805 // fc
            ]
        );
    }
}
