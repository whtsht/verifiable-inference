pub mod compiler;
pub mod model;

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
