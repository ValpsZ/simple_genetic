use rand::Rng;
use simple_genetic::simple_genetic::{activations, agent::{self, Agent}, fitness};


#[test]
fn example_use() {
    let generations = 100;
    let num_agents: usize = 10; // Has to be even, else repoulation fails
    let mut agents: Vec<Agent> =
        agent::create_agents(num_agents, 20, 1, 10, 1, 0.002, activations::TANH, activations::TANH, fitness::MEAN_SQUARED);

    for _ in 0..generations {
        let mut rng = rand::thread_rng();
        let rand_f32: f32 = rng.gen_range(0.0..1.0);
        let input = f32::sin(rand_f32);
        for idx in 0..agents.len() {
            agents[idx].set_value(input, 0, 0);
            agents[idx].calculate();
            agents[idx].calculate_fitness(vec![rand_f32]);
        }

        agents.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        agents.drain((num_agents / 2)..);


    }
}