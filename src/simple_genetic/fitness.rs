/// Function to calculate the mean squared error between two vectors.
pub const MEAN_SQUARED: &dyn Fn(Vec<f32>, Vec<f32>) -> f32 = &|output, expected| {
    assert_eq!(
        output.len(),
        expected.len(),
        "Vectors must have the same length"
    );

    let len = output.len() as f32;

    // Calculate the mean squared error
    let result: f32 = output
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e).powi(2))
        .sum();
    result / len
};
