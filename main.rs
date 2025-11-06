use std::collections::HashMap;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OpType {
    Annihilation,
    Creation
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Operator {
    op: OpType,
    index: usize,
}

impl Ord for Operator {
    /// Rule 1: Creation < Annihilation (so creations are "to the left")
    /// Rule 2: within the same kind:
    /// - creations sort by index ascending
    /// - annihilations sort by index descending
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.op, other.op) {
            (OpType::Creation, OpType::Annihilation)     => Ordering::Less,
            (OpType::Annihilation, OpType::Creation)     => Ordering::Greater,
            (OpType::Creation, OpType::Creation)         => self.index.cmp(&other.index),
            (OpType::Annihilation, OpType::Annihilation) => other.index.cmp(&self.index),
        }
    }
}

impl PartialOrd for Operator {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Term {
    coeff: i64,
    ops: Vec<Operator>,
}

#[derive(Clone, PartialEq, Eq)]
struct Expression {
    terms: Vec<Term>,
}

impl std::fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op {
            OpType::Annihilation => write!(f, "a({})", self.index),
            OpType::Creation     => write!(f, "a+({})", self.index),
        }
    }
}

impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} × {:?}", self.coeff, self.ops)
    }
}

fn creation(index: usize) -> Operator {
    Operator { op: OpType::Creation, index }
}

fn annihilation(index: usize) -> Operator {
    Operator { op: OpType::Annihilation, index }
}

/// Rule for commutation:
/// - Two creation operators always commute
/// - Two annihilation operators always commute
/// - A creation and an annihilation operator commute if they act on different indices
fn commutes(op1: &Operator, op2: &Operator) -> bool {
    match (op1.op, op2.op) {
        (OpType::Creation, OpType::Creation)         => true,
        (OpType::Annihilation, OpType::Annihilation) => true,
        (OpType::Creation, OpType::Annihilation)     => op1.index != op2.index,
        (OpType::Annihilation, OpType::Creation)     => op1.index != op2.index,
    }
}

fn normal_order_many(terms: Vec<Term>) -> Vec<Term> {
    let mut result: Vec<Term> = Vec::new();
    for term in terms {
        let mut normal_terms = normal_order(term);
        result.append(&mut normal_terms);
    }
    consolidate(result)
}

fn normal_order(term: Term) -> Vec<Term> {
    let mut queue = vec![term];
    let mut result: Vec<Term> = Vec::new();

    'main_loop: while let Some(mut term) = queue.pop() {
        for i in 1..term.ops.len() {
            let mut j = i;
            while j > 0 && term.ops[j] < term.ops[j-1] {
                if commutes(&term.ops[j], &term.ops[j-1]) {
                    term.ops.swap(j, j - 1);
                    j -= 1;
                } else {
                    let index = j - 1;
                    let prefix = &term.ops[..index];
                    let suffix = &term.ops[index + 2..];

                    // Contraction
                    let mut contracted_ops = Vec::with_capacity(prefix.len() + suffix.len());
                    contracted_ops.extend_from_slice(prefix);
                    contracted_ops.extend_from_slice(suffix);
                    queue.push(Term { coeff: term.coeff, ops: contracted_ops });
                    
                    // Swap
                    let mut swapped_ops = term.ops.clone();
                    swapped_ops.swap(index, index + 1);
                    queue.push(Term { coeff: term.coeff, ops: swapped_ops });
                    continue 'main_loop;
                }
            }
        }
        result.push(term);
    }

    consolidate(result)
}

fn consolidate(terms: Vec<Term>) -> Vec<Term> {
    let mut consolidated: HashMap<Vec<Operator>, i64> = HashMap::new();

    for term in terms {
        *consolidated.entry(term.ops).or_insert(0) += term.coeff;
    }

    let mut result: Vec<Term> = consolidated
        .into_iter()
        .filter(|(_, c)| *c != 0)
        .map(|(ops, coeff)| Term { coeff, ops })
        .collect();
    
    result.sort_by(|a, b| a.ops.len().cmp(&b.ops.len()));
    result
}

fn main() {
    let input = Term {
        coeff: 1, 
        ops: vec![
            // creation(1),
            annihilation(1),
            creation(1),
    ]};

    let result = normal_order(input.clone());
    println!("Input: {:?}", input);
    println!("Output:");
    for term in result {
        println!("{:?}", term);
    }

    let input2 = Term {coeff: 1, 
        ops: vec![
        annihilation(1),
        annihilation(1),
        creation(1),
        creation(1),
    ]};

    println!("\nInput 2: {:?}", input2);
    let result2 = normal_order(input2);
    for term in result2 {
        println!("{:?}", term);
    }


    let input3 = vec![
        Term {coeff: 1, 
        ops: vec![
            annihilation(2),
            creation(1),
        ]},
        Term {coeff: 1, 
        ops: vec![
            creation(1),
            annihilation(2),
        ]},
    ];

    println!("\nInput 3: {:?}", input3);
    let result3 = normal_order_many(input3);
    for term in result3 {
        println!("{:?}", term);
    }
}
