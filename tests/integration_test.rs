#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use simple_genetic::simple_genetic::{activations, agent::create_agents, fitness};

    #[test]
    fn example_use() {
        // Constants
        const GENERATIONS: usize = 100;
        const NUM_AGENTS: usize = 10;
        const GENOME_LENGTH: usize = 20;
        const INPUT_SIZE: u32 = 1;
        const HIDDEN_SIZE: u32 = 10;
        const OUTPUT_SIZE: u32 = 1;
        const MUTATION_RATE: f64 = 0.002;

        // Create agents
        let mut agents = create_agents(
            NUM_AGENTS, // Initial population size
            GENOME_LENGTH,
            INPUT_SIZE,
            HIDDEN_SIZE,
            OUTPUT_SIZE,
            MUTATION_RATE,
            activations::TANH,
            activations::TANH,
            fitness::MEAN_SQUARED,
        );

        // Training loop
        for generation in 0..GENERATIONS {
            let input_value = thread_rng().gen_range(-std::f32::consts::PI..std::f32::consts::PI);
            let expected_output = vec![input_value.sin()];

            for agent in &mut agents {
                agent.set_value(input_value, 0, 0);
                agent.calculate();
                agent.calculate_fitness(expected_output.clone());
            }

            agents.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            let best_fittness = agents[0].fitness;
            let selected_agents = agents
                .iter()
                .take(NUM_AGENTS / 2)
                .cloned()
                .collect::<Vec<_>>();

            let mut new_population = Vec::new();
            for idx in 0..NUM_AGENTS {
                let parent = &selected_agents[idx];
                let child = parent.reproduce();
                new_population.push(child);
            }
            agents = new_population;

            println!("Generation {}: Best fitness: {}", generation, best_fittness);
        }
        agents[0].genome.print_brain();
    }
}
