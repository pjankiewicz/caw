use rand::Rng;
use rand::seq::index::sample;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct CGPGenotype {
    pub function_genes: Vec<usize>,
    pub connection_genes: Vec<Vec<usize>>,
    pub output_genes: Vec<usize>,
    pub num_inputs: usize,
    pub num_nodes: usize,       // Now calculated as rows * columns
    pub num_outputs: usize,
    pub max_arity: usize,
    pub rows: usize,
    pub columns: usize,
    pub levels_back: usize,
    pub num_functions: usize,
}


impl CGPGenotype {
    /// Initializes a new CGP genotype.
    pub fn new<R: Rng + ?Sized>(
        rng: &mut R,
        num_inputs: usize,
        num_outputs: usize,
        max_arity: usize,
        rows: usize,
        columns: usize,
        levels_back: usize,
        num_functions: usize,
    ) -> Self {
        let num_nodes = rows * columns;
        let mut function_genes = Vec::with_capacity(num_nodes);
        let mut connection_genes = Vec::with_capacity(num_nodes);

        let mut genotype = CGPGenotype {
            function_genes: Vec::new(),
            connection_genes: Vec::new(),
            output_genes: Vec::new(),
            num_inputs,
            num_nodes,
            num_outputs,
            max_arity,
            rows,
            columns,
            levels_back,
            num_functions,
        };

        for node_index in 0..num_nodes {
            // Randomly select a function
            let function_gene = rng.gen_range(0..num_functions);
            function_genes.push(function_gene);

            // Get allowed connections
            let allowed_indices = genotype.allowed_connection_indices(node_index);

            // Initialize connections with max arity
            let inputs = (0..max_arity)
                .map(|_| {
                    let idx = rng.gen_range(0..allowed_indices.len());
                    allowed_indices[idx]
                })
                .collect();
            connection_genes.push(inputs);
        }

        // Initialize output genes
        let total_nodes = num_inputs + num_nodes;
        let output_genes = (0..num_outputs)
            .map(|_| rng.gen_range(num_inputs..total_nodes))
            .collect();

        genotype.function_genes = function_genes;
        genotype.connection_genes = connection_genes;
        genotype.output_genes = output_genes;

        genotype
    }

    /// Computes the allowed connection range for a node based on levels_back.
    /// Computes the allowed connection range for a node based on levels_back.
    fn allowed_connection_indices(&self, node_index: usize) -> Vec<usize> {
        let node_column = node_index / self.rows;
        let levels_back = self.levels_back.min(node_column);

        let min_column = node_column.saturating_sub(levels_back);
        let min_node_index = min_column * self.rows;
        let max_node_index = node_index;

        let mut allowed_indices = Vec::new();

        // Include inputs
        allowed_indices.extend(0..self.num_inputs);

        // Include nodes within levels_back
        for idx in min_node_index..max_node_index {
            allowed_indices.push(self.num_inputs + idx);
        }

        allowed_indices
    }

    /// Mutates the genotype with a given number of mutations.
    pub fn mutate<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        num_mutations: usize,
    ) {
        let total_connection_genes = self.connection_genes.len() * self.max_arity;
        let total_genes = self.function_genes.len() + total_connection_genes + self.output_genes.len();

        let mutation_indices = sample(rng, total_genes, num_mutations).into_vec();

        for gene_idx in mutation_indices {
            if gene_idx < self.function_genes.len() {
                // Mutate function gene
                self.function_genes[gene_idx] = rng.gen_range(0..self.num_functions);
            } else if gene_idx < self.function_genes.len() + total_connection_genes {
                // Mutate connection gene
                let conn_gene_idx = gene_idx - self.function_genes.len();
                let node_idx = conn_gene_idx / self.max_arity;
                let input_idx = conn_gene_idx % self.max_arity;

                let allowed_indices = self.allowed_connection_indices(node_idx);
                let new_connection = allowed_indices[rng.gen_range(0..allowed_indices.len())];
                self.connection_genes[node_idx][input_idx] = new_connection;
            } else {
                // Mutate output gene
                let output_idx = gene_idx - self.function_genes.len() - total_connection_genes;
                let total_nodes = self.num_inputs + self.num_nodes;
                self.output_genes[output_idx] = rng.gen_range(self.num_inputs..total_nodes);
            }
        }
    }

    /// Identifies active nodes needed for evaluation.
    pub fn active_nodes(&self) -> HashSet<usize> {
        let mut to_evaluate = vec![false; self.num_nodes];
        let mut nodes_to_process = Vec::new();

        // Mark output genes for evaluation
        for &output_gene in &self.output_genes {
            if output_gene >= self.num_inputs {
                let node_idx = output_gene - self.num_inputs;
                if !to_evaluate[node_idx] {
                    to_evaluate[node_idx] = true;
                    nodes_to_process.push(node_idx);
                }
            }
        }

        // Backward traversal to find all active nodes
        while let Some(node_idx) = nodes_to_process.pop() {
            for &input_idx in &self.connection_genes[node_idx] {
                if input_idx >= self.num_inputs {
                    let input_node_idx = input_idx - self.num_inputs;
                    if !to_evaluate[input_node_idx] {
                        to_evaluate[input_node_idx] = true;
                        nodes_to_process.push(input_node_idx);
                    }
                }
            }
        }

        // Collect indices of active nodes
        to_evaluate
            .iter()
            .enumerate()
            .filter_map(|(idx, &active)| if active { Some(idx) } else { None })
            .collect()
    }
}

/// Evolves a CGP genotype using an evolutionary strategy (1 + λ).
/// - `fitness_fn`: Function to evaluate the fitness of a genotype.
/// - `num_generations`: Number of generations to run.
/// - `num_mutations`: Number of mutations per offspring.
/// - `lambda`: Number of offspring per generation.
/// Returns the best genotype found.
pub fn evolve<R, F>(
    rng: &mut R,
    initial_genotype: CGPGenotype,
    fitness_fn: F,
    num_generations: usize,
    num_mutations: usize,
    lambda: usize
) -> CGPGenotype
    where
        R: Rng + ?Sized,
        F: Fn(&CGPGenotype) -> f64,
{
    let mut parent_genotype = initial_genotype;
    let mut best_fitness = fitness_fn(&parent_genotype);

    for generation in 0..num_generations {
        let mut offspring_genotypes = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            let mut offspring = parent_genotype.clone();
            offspring.mutate(rng, num_mutations);
            offspring_genotypes.push(offspring);
        }

        for offspring in offspring_genotypes {
            let offspring_fitness = fitness_fn(&offspring);
            if offspring_fitness <= best_fitness {
                parent_genotype = offspring;
                best_fitness = offspring_fitness;
            }
        }

        println!("Generation {} Best fitness: {}", generation, best_fitness);
    }

    parent_genotype
}
