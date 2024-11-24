use bincode::{deserialize, serialize};
use std::fs::File;
use std::io::{Read, Write};

pub fn save_to_file<T: serde::Serialize>(map: &T, filename: &str) -> std::io::Result<()> {
    let encoded: Vec<u8> = serialize(map).unwrap();
    let mut file = File::create(filename)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load_from_file<'a, T: serde::Deserialize<'a>>(
    filename: &str,
    buffer: &'a mut Vec<u8>,
) -> std::io::Result<T> {
    let mut file = File::open(filename)?;
    file.read_to_end(buffer)?;
    Ok(deserialize(buffer).unwrap())
}
