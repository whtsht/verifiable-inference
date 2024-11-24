use curve25519_dalek::Scalar;

pub fn from_i64(x: i64) -> Scalar {
    let x_abs = x.unsigned_abs();
    if x < 0 {
        -Scalar::from(x_abs)
    } else {
        Scalar::from(x_abs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_i64() {
        assert_eq!(from_i64(1234), Scalar::from(1234u32));
        assert_eq!(from_i64(-1234), -Scalar::from(1234u32));
    }
}
