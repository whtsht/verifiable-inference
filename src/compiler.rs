// model
// input [a b
//        c d]
// conv  [x] <-
// conv  [y]
// output [e f
//         g h]
//
// expression
//
// a * x * y = e,
// b * x * y = f,
// c * x * y = g,
// d * x * y = h,
//
// Flattening (Algebraic Circuit)
//
// $1 = a  * x
// e  = $1 * y
// $1 = b  * x
// f  = $1 * y
// ...
//
// c * x * y = g
// d * x * y = h
//
//
// constructs
// inputs
// veriables
//
// Our R1CS instance is three constraints over five variables and two public inputs
// (Z0 + Z1) * I0 - Z2 = 0
// (Z0 + I1) * Z2 - Z3 = 0
// Z4 * 1 - 0 = 0

pub type Id = usize;

pub enum Circuit {
    // x = a
    Eq(Id, Id),
    // x = a + b
    Mult(Id, Id, Id),
    // x = a * b
    Add(Id, Id, Id),
}

// Client, Worker, TrustedParty
//
// Circuit(input) -> output
// R1CS: Circuit(input) = output

// 1. TrustedParty
// Model -> Circuit -> R1CS
// R1CS  -> Gens
// R1CS, Gens -> Commitment
//
// 2. Worker
// TrustedParty -> Circuit
// TrustedParty -> R1CS
// TrustedParty -> Gens
//
// 3. Client
// TrustedParty -> Gens, Commitment
//
// 4. Worker
// Client -> Input
// Circuit, Input -> Output
// R1CS, Input, Output -> Proof
//
// 5. Client
// Worker -> Proof, Output
// Commitment, Gens, Input, Output, Proof -> Verify
