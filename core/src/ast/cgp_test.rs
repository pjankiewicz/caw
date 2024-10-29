use crate::ast::cgp::CGPGenotype;

type Function = fn(&[f64]) -> f64;

struct FunctionSet {
    functions: Vec<(Function, usize)>, // (function, arity)
}

impl FunctionSet {
    fn new() -> Self {
        let functions: Vec<(Function, usize)> = vec![
            (|inputs: &[f64]| inputs[0] + inputs[1], 2), // Addition
            (|inputs: &[f64]| inputs[0] - inputs[1], 2), // Subtraction
            (|inputs: &[f64]| inputs[0] * inputs[1], 2), // Multiplication
            (|inputs: &[f64]| if inputs[1] != 0.0 { inputs[0] / inputs[1] } else { 0.0 }, 2), // Division with zero check
            (|inputs: &[f64]| inputs[0], 1),             // Identity function
            (|_: &[f64]| 1.0, 0),                        // Constant 1
            (|_: &[f64]| 2.0, 0),                        // Constant 2
        ];
        Self { functions }
    }

    fn get_function(&self, index: usize) -> &(Function, usize) {
        &self.functions[index]
    }

    fn len(&self) -> usize {
        self.functions.len()
    }

    fn max_arity(&self) -> usize {
        self.functions.iter().map(|&(_, arity)| arity).max().unwrap_or(0)
    }
}

fn evaluate_genotype(
    genotype: &CGPGenotype,
    inputs: &[f64],
    function_set: &FunctionSet,
) -> Vec<f64> {
    let num_inputs = genotype.num_inputs;
    let total_nodes = num_inputs + genotype.num_nodes;
    let mut node_outputs = vec![0.0; total_nodes];

    // Load input data
    for i in 0..num_inputs {
        node_outputs[i] = inputs[i];
    }

    // Identify active nodes
    let active_nodes = genotype.active_nodes();

    // Evaluate active nodes
    let mut execution_order: Vec<usize> = active_nodes.into_iter().collect();
    execution_order.sort_unstable(); // Ensure proper order

    for node_idx in execution_order {
        let f_idx = genotype.function_genes[node_idx];
        let (function, arity) = function_set.get_function(f_idx);

        let input_indices = &genotype.connection_genes[node_idx][..*arity];
        let input_values: Vec<f64> = input_indices.iter().map(|&idx| node_outputs[idx]).collect();

        node_outputs[num_inputs + node_idx] = function(&input_values);
    }

    // Collect outputs
    genotype
        .output_genes
        .iter()
        .map(|&idx| node_outputs[idx])
        .collect()
}

fn genotype_to_formula(genotype: &CGPGenotype, function_set: &FunctionSet) -> String {
    let num_inputs = genotype.num_inputs;

    // Recursive function to build a formula for each node
    fn node_formula(
        node_idx: usize,
        genotype: &CGPGenotype,
        function_set: &FunctionSet,
        node_outputs: &mut Vec<Option<String>>,
        num_inputs: usize,
    ) -> String {
        // If the node output has already been computed, return it
        if let Some(formula) = &node_outputs[node_idx] {
            return formula.clone();
        }

        // If this is an input node, return it directly
        if node_idx < num_inputs {
            let input_var = format!("x{}", node_idx);
            node_outputs[node_idx] = Some(input_var.clone());
            return input_var;
        }

        // Otherwise, it's a computational node
        let f_idx = genotype.function_genes[node_idx - num_inputs];
        let (function, arity) = function_set.get_function(f_idx);

        // Collect input formulas recursively
        let input_indices = &genotype.connection_genes[node_idx - num_inputs][..*arity];
        let input_formulas: Vec<String> = input_indices
            .iter()
            .map(|&idx| node_formula(idx, genotype, function_set, node_outputs, num_inputs))
            .collect();

        // Build the formula based on the function
        let formula = match f_idx {
            0 => format!("({} + {})", input_formulas[0], input_formulas[1]),  // Addition
            1 => format!("({} - {})", input_formulas[0], input_formulas[1]),  // Subtraction
            2 => format!("({} * {})", input_formulas[0], input_formulas[1]),  // Multiplication
            3 => format!("({} / {})", input_formulas[0], input_formulas[1]),  // Division
            4 => format!("{}", input_formulas[0]),                            // Identity
            5 => "1.0".to_string(),                                          // Constant 1
            6 => "2.0".to_string(),                                          // Constant 2
            _ => "unknown_function".to_string(),                             // Unknown function
        };

        // Store and return the formula for this node
        node_outputs[node_idx] = Some(formula.clone());
        formula
    }

    // Initialize storage for node formulas
    let total_nodes = num_inputs + genotype.num_nodes;
    let mut node_outputs: Vec<Option<String>> = vec![None; total_nodes];

    // Compute the formula for each output gene
    genotype
        .output_genes
        .iter()
        .map(|&idx| node_formula(idx, genotype, function_set, &mut node_outputs, num_inputs))
        .collect::<Vec<_>>()
        .join(", ")
}


fn fitness_fn(genotype: &CGPGenotype, function_set: &FunctionSet) -> f64 {
    // Define a set of test inputs
    let test_inputs = vec![
        -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
    ];

    let mut total_error = 0.0;

    for &x in &test_inputs {
        // Evaluate the genotype with input x
        let outputs = evaluate_genotype(genotype, &[x], function_set);

        // Target output
        let target_output = x * x * x + x * x + 2.0 * x + 1.0;

        // Calculate squared error
        let error = outputs[0] - target_output;
        total_error += error * error;
    }

    total_error // Lower is better
}

#[cfg(test)]
mod tests {
    use crate::ast::cgp::{CGPGenotype, evolve};
    use crate::ast::cgp_test::{evaluate_genotype, fitness_fn, FunctionSet, genotype_to_formula};

    #[test]
    fn main() {
        let num_inputs = 1;        // Only x
        let num_outputs = 1;       // Output of the expression
        let num_nodes = 50;        // Number of computational nodes
        let rows = 2;              // Single row
        let columns = 5;          // Number of columns equal to number of nodes (since rows = 1)
        let levels_back = columns; // Allow connections to any previous node
        let lambda = 4;            // Number of offspring per generation
        let num_generations = 10000; // Increased for better convergence
        let num_mutations = 5;

        // Initialize function set
        let function_set = FunctionSet::new();
        let num_functions = function_set.len();
        let max_arity = function_set.max_arity();

        let mut rng = rand::thread_rng();

        // Initialize the genotype
        let initial_genotype = CGPGenotype::new(
            &mut rng,
            num_inputs,
            num_outputs,
            max_arity,
            rows,
            columns,
            levels_back,
            num_functions,
        );

        // Evolution loop
        let best_genotype = evolve(
            &mut rng,
            initial_genotype,
            |genotype| fitness_fn(genotype, &function_set),
            num_generations,
            num_mutations,
            lambda
        );

        // Test the best genotype
        let test_inputs = vec![-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
                               -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];

        println!("Evolved expression outputs:");

        for &x in &test_inputs {
            let outputs = evaluate_genotype(&best_genotype, &[x], &function_set);
            let target_output = x * x * x + x * x + 2.0 * x + 1.0;
            println!("x = {:>5.2}, CGP Output = {:>8.4}, Target = {:>8.4}, Error = {:>8.4}",
                     x, outputs[0], target_output, (outputs[0] - target_output).abs());
        }

        let formula = genotype_to_formula(&best_genotype, &function_set);
        println!("Evolved expression formula: {}", formula);
    }
}
