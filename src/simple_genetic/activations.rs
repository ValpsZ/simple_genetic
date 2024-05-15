use std::f32::consts::E;

/// Hyperbolic tangent activation function.
pub const TANH: &dyn Fn(f32) -> f32 = &|x| (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x));

/// Linear activation function.
pub const LIN: &dyn Fn(f32) -> f32 = &|x| (x);
