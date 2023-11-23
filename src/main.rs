use rand::prelude::*;

struct Agent {
    genome: Genome,
    input_list: Vec<f32>,
    hidden_list: Vec<f32>,
    output_list: Vec<f32>,
    input_neurons: u32,
    hidden_neurons: u32,
    output_neurons: u32,
    mutation_rate: f64,
}

struct Genome {
    gene_list: Vec<u32>,
    input_neurons: u32,
    hidden_neurons: u32,
    output_neurons: u32,
}

impl Genome {
    fn print_brain(&self) -> () {
        println!("Input neurons: {}", &self.input_neurons);
        println!("Hidden neurons: {}", &self.hidden_neurons);
        println!("Output neurons: {}", &self.output_neurons);
        for i in 0..(&self.gene_list).len() {
            let gene_number = &self.gene_list[i];
            println!("{}: Gene {gene_number}", i + 1);
            if (&self.gene_list[i] & (0x1 << 31)) != 0 {
                let hidden_neuron_index =
                    (&self.gene_list[i] & (0x01111111 << 24)) % &self.hidden_neurons;
                if (&self.gene_list[i] & (0x000000001 << 23)) != 0 {
                    let output_neuron_index =
                        (&self.gene_list[i] & (0x0000000001111111 << 16)) % &self.output_neurons;
                    println!("{}: Hidden neuron number {hidden_neuron_index} to output neuron number {output_neuron_index}", i + 1);
                } else {
                    let hidden_neuron_index_2 =
                        (&self.gene_list[i] & (0x0000000001111111 << 16)) % &self.hidden_neurons;
                    println!("{}", (&self.gene_list[i] & (0x0000000001111111 << 16)));
                    println!("{}: Hidden neuron number {hidden_neuron_index} to hidden neuron number {hidden_neuron_index_2}", i + 1);
                }
            } else {
                let input_neuron_index =
                    (&self.gene_list[i] & (0x01111111 << 24)) % &self.input_neurons;
                if (&self.gene_list[i] & (0x000000001 << 23)) != 0 {
                    let output_neuron_index =
                        (&self.gene_list[i] & (0x0000000001111111 << 16)) % &self.output_neurons;
                    println!("{}: Input neuron number {input_neuron_index} to output neuron number {output_neuron_index}", i + 1);
                } else {
                    let hidden_neuron_index =
                        (&self.gene_list[i] & (0x0000000001111111 << 16)) % &self.hidden_neurons;
                    println!("{}: Input neuron number {input_neuron_index} to hidden neuron number {hidden_neuron_index}", i + 1);
                }
            }

            let weight;
            if (&self.gene_list[i] & 0b00000000000000001000000000000000u32) == 0 {
                weight =
                    (&self.gene_list[i] & 0b00000000000000000111111111111111u32) as f32 / 6000.0;
            } else {
                weight =
                    -((&self.gene_list[i] & 0b00000000000000000111111111111111u32) as f32 / 6000.0);
            }
            println!("{}: Weight {weight}", i + 1)
        }
    }
}

impl Clone for Genome {
    fn clone(&self) -> Self {
        Self {
            gene_list: self.gene_list.clone(),
            input_neurons: self.input_neurons,
            hidden_neurons: self.hidden_neurons,
            output_neurons: self.output_neurons,
        }
    }
}

impl Agent {
    fn calculate(&mut self) -> () {
        for i in 0..(&self.genome.gene_list).len() {
            let first_layer_index;
            let second_layer_index;
            let first_neuron_index;
            let second_neuron_index;
            if (&self.genome.gene_list[i] & (0x1 << 31)) != 0 {
                first_layer_index = 1;
                first_neuron_index =
                    (&self.genome.gene_list[i] & (0x01111111 << 24)) % &self.hidden_neurons;
            } else {
                first_layer_index = 0;
                first_neuron_index =
                    (&self.genome.gene_list[i] & (0x01111111 << 24)) % &self.input_neurons;
            }

            if (&self.genome.gene_list[i] & (0x000000001 << 23)) != 0 {
                second_layer_index = 2;
                second_neuron_index =
                    (&self.genome.gene_list[i] & (0x0000000001111111 << 16)) % &self.output_neurons;
            } else {
                second_layer_index = 1;
                second_neuron_index =
                    (&self.genome.gene_list[i] & (0x0000000001111111 << 16)) % &self.hidden_neurons;
            }

            let weight;
            if (&self.genome.gene_list[i] & 0b00000000000000001000000000000000u32) == 0 {
                weight = (&self.genome.gene_list[i] & 0b00000000000000000111111111111111u32) as f32
                    / 6000.0;
            } else {
                weight = -((&self.genome.gene_list[i] & 0b00000000000000000111111111111111u32)
                    as f32
                    / 6000.0);
            }

            if first_layer_index == 0 {
                self.set_value(
                    &self.input_list[first_neuron_index as usize] * weight,
                    second_neuron_index as usize,
                    second_layer_index,
                );
            } else {
                self.set_value(
                    &self.hidden_list[first_neuron_index as usize] * weight,
                    second_neuron_index as usize,
                    second_layer_index,
                );
            }
        }
    }

    fn set_value(&mut self, value: f32, index: usize, list_index: i32) -> () {
        match list_index {
            0 => self.input_list[index] = value,
            1 => self.hidden_list[index] = value,
            2 => self.output_list[index] = value,
            _ => println!("Index not valid"),
        }
    }

    fn mutation_u32(&self) -> u32 {
        let mut rng = rand::thread_rng();

        let mut result: u32 = 0;
        for i in 0..31 {
            let rand: f64 = rng.gen();
            if rand < self.mutation_rate {
                result = result | 0x1 << i;
            }
        }

        result
    }

    fn reproduce(&self) -> Agent {
        let mut new_gene_list: Vec<u32> = vec![];
        for i in 0..(&self.genome.gene_list).len() {
            new_gene_list.push(&self.genome.gene_list[i] ^ self.mutation_u32());
        }

        Agent {
            genome: Genome {
                gene_list: new_gene_list,
                input_neurons: self.input_neurons,
                hidden_neurons: self.hidden_neurons,
                output_neurons: self.output_neurons,
            },
            input_list: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            hidden_list: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            output_list: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            input_neurons: self.input_neurons,
            hidden_neurons: self.hidden_neurons,
            output_neurons: self.output_neurons,
            mutation_rate: self.mutation_rate,
        }
    }
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Self {
            genome: self.genome.clone(),
            input_list: self.input_list.clone(),
            hidden_list: self.hidden_list.clone(),
            output_list: self.output_list.clone(),
            input_neurons: self.input_neurons,
            hidden_neurons: self.hidden_neurons,
            output_neurons: self.output_neurons,
            mutation_rate: self.mutation_rate,
        }
    }
}

fn create_agents(amount: usize, genome_length: usize, input_size: u32, hidden_size: u32, output_size: u32, mutation_rate: f64) -> Vec<Agent> {
    let mut result: Vec<Agent> = Vec::with_capacity(amount);
    for _ in 0..amount {
        let new_agent = Agent {
            genome: Genome{
                gene_list: vec![0; genome_length],
                input_neurons: input_size,
                hidden_neurons: hidden_size,
                output_neurons: output_size,
            },
            input_list: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            hidden_list: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            output_list: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            input_neurons: input_size,
            hidden_neurons: hidden_size,
            output_neurons: output_size,
            mutation_rate: mutation_rate,
        };
        result.push(new_agent)
    }
    result
}

fn fitness(result: Vec<f32>, expected: &Vec<f32>) -> f32 {
    result.into_iter().zip(expected).map(|(r, e)| f32::powf(r-e,2.0)).sum::<f32>()
}

fn main() {
    let mut agent_list: Vec<Agent> = create_agents(10, 5, 1, 5, 5, 0.01);
    let goal: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let mut best_fittnes: f32 = 100.0;

    for generation in 0..40 {
        println!("Generation: {generation}");  

        let new_agent_list: Vec<Agent> = agent_list.iter()
                                                    .map(|x| x.reproduce())
                                                    .collect::<Vec<Agent>>();
        
        agent_list = new_agent_list;

        for i in 0..agent_list.len() {
            agent_list[i].set_value(1.0, 0, 0);
        }

        let mut fittnes_list: Vec<f32> = Vec::with_capacity(agent_list.len());

        for i in 0..agent_list.len() {
            agent_list[i].calculate();
            let f: f32 = fitness(agent_list[i].output_list.clone(), &goal);
            if f < best_fittnes {
                agent_list[i].genome.print_brain();
                best_fittnes = f;
            }
            println!("Agent: {i} Fittnes: {f}");
            fittnes_list.push(f);
        }

        let avg_fittness: f32 = fittnes_list.iter().sum::<f32>() / fittnes_list.len() as f32;
        println!("Average fittness: {avg_fittness}");
        
        for i in (0..(agent_list.len()-1)).rev() {
            if fittnes_list[i] > avg_fittness {
                agent_list.remove(i);
            }
        }


        while agent_list.len() < fittnes_list.len() {
            agent_list.push(agent_list[(rand::random::<f32>() * agent_list.len() as f32).floor() as usize].clone());
        }
    }
    println!("{:?}", agent_list[0].output_list);
}