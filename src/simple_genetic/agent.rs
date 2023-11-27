use rand::prelude::*;

pub struct Agent {
    genome: Genome,
    input_list: Vec<f32>,
    hidden_list: Vec<f32>,
    output_list: Vec<f32>,
    input_neurons: u32,
    hidden_neurons: u32,
    output_neurons: u32,
    mutation_rate: f64,
    fitness: f32,
}

pub struct Genome {
    gene_list: Vec<u32>,
    input_neurons: u32,
    hidden_neurons: u32,
    output_neurons: u32,
}

impl Genome {
    pub fn print_brain(&self) -> () {
        println!("Input neurons: {}", &self.input_neurons);
        println!("Hidden neurons: {}", &self.hidden_neurons);
        println!("Output neurons: {}", &self.output_neurons);
        for i in 0..((&self.gene_list).len()) {
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
    pub fn calculate(&mut self) -> () {
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
                if second_layer_index == 1 {
                    self.set_value(
                        &self.hidden_list[second_neuron_index as usize]
                            + (&self.input_list[first_neuron_index as usize] * weight),
                        second_neuron_index as usize,
                        second_layer_index,
                    );
                } else {
                    self.set_value(
                        &self.output_list[second_neuron_index as usize]
                            + (&self.input_list[first_neuron_index as usize] * weight),
                        second_neuron_index as usize,
                        second_layer_index,
                    );
                }
            } else {
                if second_layer_index == 1 {
                    self.set_value(
                        &self.hidden_list[second_neuron_index as usize]
                            + (&self.hidden_list[first_neuron_index as usize] * weight),
                        second_neuron_index as usize,
                        second_layer_index,
                    );
                } else {
                    self.set_value(
                        &self.output_list[second_neuron_index as usize]
                            + (&self.hidden_list[first_neuron_index as usize] * weight),
                        second_neuron_index as usize,
                        second_layer_index,
                    );
                }
            }
        }
    }

    pub fn set_value(&mut self, value: f32, index: usize, list_index: i32) -> () {
        match list_index {
            0 => self.input_list[index] = value,
            1 => self.hidden_list[index] = value,
            2 => self.output_list[index] = value,
            _ => panic!("Index ({index}) not valid"),
        }
    }

    pub fn clear_values(&mut self) -> () {
        self.input_list = vec![0.0; self.input_list.len()];
        self.hidden_list = vec![0.0; self.hidden_list.len()];
        self.output_list = vec![0.0; self.output_list.len()];
    }

    pub fn mutation_u32(&self) -> u32 {
        let mut rng = rand::thread_rng();

        let mut result: u32 = 0;
        for i in 0..32 {
            let rand: f64 = rng.gen();
            if rand < self.mutation_rate {
                result = result | 0x1 << i;
            }
        }

        result
    }

    pub fn reproduce(&self) -> Agent {
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
            input_list: vec![0.0; self.input_neurons as usize],
            hidden_list: vec![0.0; self.hidden_neurons as usize],
            output_list: vec![0.0; self.output_neurons as usize],
            input_neurons: self.input_neurons,
            hidden_neurons: self.hidden_neurons,
            output_neurons: self.output_neurons,
            mutation_rate: self.mutation_rate,
            fitness: self.fitness,
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
            fitness: self.fitness,
        }
    }
}

fn create_agents(
    amount: usize,
    genome_length: usize,
    input_size: u32,
    hidden_size: u32,
    output_size: u32,
    mutation_rate: f64,
) -> Vec<Agent> {
    let mut result: Vec<Agent> = Vec::with_capacity(amount);
    for _ in 0..amount {
        let new_agent = Agent {
            genome: Genome {
                gene_list: vec![0; genome_length],
                input_neurons: input_size,
                hidden_neurons: hidden_size,
                output_neurons: output_size,
            },
            input_list: vec![0.0; input_size as usize],
            hidden_list: vec![0.0; hidden_size as usize],
            output_list: vec![0.0; output_size as usize],
            input_neurons: input_size,
            hidden_neurons: hidden_size,
            output_neurons: output_size,
            mutation_rate: mutation_rate,
            fitness: 100.0,
        };
        result.push(new_agent)
    }
    result
}
