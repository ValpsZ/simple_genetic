use std::f32::consts::E;

pub const TANH: &dyn Fn(f32) -> f32 = &|x| (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x));
pub const LIN: &dyn Fn(f32) -> f32 = &|x| (x);
