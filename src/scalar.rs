use curve25519_dalek::Scalar;

pub fn from_i32(x: i32) -> Scalar {
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
    fn test_from_i32() {
        assert_eq!(from_i32(1234), Scalar::from(1234u32));
        assert_eq!(from_i32(-1234), -Scalar::from(1234u32));
    }
}
