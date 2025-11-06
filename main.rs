use std::collections::HashMap;
use std::cmp::Ordering;
use std::ops::{Add, Mul};

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

impl Operator {
    fn creation(index: usize) -> Operator {
        Operator { op: OpType::Creation, index }
    }

    fn annihilation(index: usize) -> Operator {
        Operator { op: OpType::Annihilation, index }
    }
}

impl std::fmt::Debug for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op {
            OpType::Annihilation => write!(f, "a({})", self.index),
            OpType::Creation     => write!(f, "a+({})", self.index),
        }
    }
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

#[derive(Clone, PartialEq)]
struct Term {
    coeff: f32,
    ops: Vec<Operator>,
}

impl Term {
    fn one_body(i: usize, j: usize) -> Term {
        Term {
            coeff: 1.0,
            ops: vec![
                Operator::creation(i),
                Operator::annihilation(j),
            ],
        }
    }

    fn two_body(i: usize, j: usize, k: usize, l: usize) -> Term {
        Term {
            coeff: 1.0,
            ops: vec![
                Operator::creation(i),
                Operator::creation(j),
                Operator::annihilation(l),
                Operator::annihilation(k),
            ],
        }
    }

    fn density(i: usize) -> Term {
        Term::one_body(i, i)
    }
}

impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} × {:?}", self.coeff, self.ops)
    }
}

#[derive(Clone, PartialEq)]
struct Expression(Vec<Term>);

impl Expression {
    fn hopping(i: usize, j: usize) -> Expression {
        Expression(vec![
            Term::one_body(i, j),
            Term::one_body(j, i),
        ])
    }
}

impl std::fmt::Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for term in &self.0 {
            writeln!(f, "{:?}", term)?;
        }
        Ok(())
    }
}

/// Implementations of Add and Mul for Term and Expression
/// Term + f32
impl Add<f32> for Term {
    type Output = Expression;

    fn add(self, rhs: f32) -> Self::Output {
        let constant_term = Term {
            coeff: rhs,
            ops: vec![],
        };
        Expression(vec![self, constant_term])
    }
}

/// f32 + Term
impl Add<Term> for f32 {
    type Output = Expression;

    fn add(self, rhs: Term) -> Self::Output {
        let constant_term = Term {
            coeff: self,
            ops: vec![],
        };
        Expression(vec![constant_term, rhs])
    }
}

/// Term + Term
impl Add<Term> for Term {
    type Output = Expression;

    fn add(self, rhs: Self) -> Self::Output {
        Expression(vec![self, rhs])
    }
}

/// Expression + f32
impl Add<f32> for Expression {
    type Output = Expression;

    fn add(self, rhs: f32) -> Self::Output {
        let mut result = self;
        let constant_term = Term {
            coeff: rhs,
            ops: vec![],
        };
        result.0.push(constant_term);
        result
    }
}

/// f32 + Expression
impl Add<Expression> for f32 {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Self::Output {
        let mut result = rhs;
        let constant_term = Term {
            coeff: self,
            ops: vec![],
        };
        result.0.push(constant_term);
        result
    }
}

/// Expression + Expression
impl Add<Expression> for Expression {
    type Output = Expression;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result.0.extend(rhs.0);
        result
    }
}

/// Expression + Term
impl Add<Term> for Expression {
    type Output = Expression;

    fn add(self, rhs: Term) -> Self::Output {
        let mut result = self;
        result.0.push(rhs);
        result
    }
}

/// Term + Expression
impl Add<Expression> for Term {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Self::Output {
        let mut result_terms = vec![self];
        result_terms.extend(rhs.0);
        Expression(result_terms)
    }
}

/// Term x f32
impl Mul<f32> for Term {
    type Output = Term;

    fn mul(self, rhs: f32) -> Self::Output {
        Term {
            coeff: self.coeff * rhs,
            ops: self.ops,
        }
    }
}

/// f32 x Term
impl Mul<Term> for f32 {
    type Output = Term;

    fn mul(self, rhs: Term) -> Self::Output {
        Term {
            coeff: rhs.coeff * self,
            ops: rhs.ops,
        }
    }
}

/// Term x Term
impl Mul<Term> for Term {
    type Output = Term;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new_ops = self.ops;
        new_ops.extend_from_slice(&rhs.ops);
        Term {
            coeff: self.coeff * rhs.coeff,
            ops: new_ops,
        }
    }
}

/// &Term x &Term -> Term 
impl<'a, 'b> Mul<&'b Term> for &'a Term {
    type Output = Term;

    fn mul(self, rhs: &'b Term) -> Self::Output {
        let mut ops = Vec::with_capacity(self.ops.len() + rhs.ops.len());
        ops.extend_from_slice(&self.ops);
        ops.extend_from_slice(&rhs.ops);
        Term { coeff: self.coeff * rhs.coeff, ops }
    }
}

/// Expression x f32
impl Mul<f32> for Expression {
    type Output = Expression;

    fn mul(self, rhs: f32) -> Self::Output {
        let new_terms: Vec<Term> = self.0.into_iter().map(|t| t * rhs).collect();
        Expression(new_terms)
    }
}

/// f32 x Expression
impl Mul<Expression> for f32 {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let new_terms: Vec<Term> = rhs.0.into_iter().map(|t| t * self).collect();
        Expression(new_terms)
    }
}

/// Expression x Term
impl Mul<Term> for Expression {
    type Output = Expression;

    fn mul(self, rhs: Term) -> Self::Output {
        let rhs_ref = &rhs;
        let new_terms: Vec<Term> = self.0.into_iter().map(|t| &t * rhs_ref).collect();
        Expression(new_terms)
    }
}

/// Term x Expression
impl Mul<Expression> for Term {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let lhs_ref = &self;
        let new_terms: Vec<Term> = rhs.0.into_iter().map(|t| lhs_ref * &t).collect();
        Expression(new_terms)
    }
}

/// Expression x 'Term
impl<'a> Mul<&'a Term> for Expression {
    type Output = Expression;

    fn mul(self, rhs: &'a Term) -> Self::Output {
        Expression(self.0.into_iter().map(|t| &t * rhs).collect())
    }
}

/// 'Term x Expression
impl<'a> Mul<&'a Expression> for Term {
    type Output = Expression;

    fn mul(self, rhs: &'a Expression) -> Self::Output {
        Expression(rhs.0.iter().map(|t| &self * t).collect())
    }
}

/// Expression x Expression
impl Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_terms = &self.0;
        let rhs_terms = &rhs.0;

        let mut result_terms = Vec::with_capacity(lhs_terms.len() * rhs_terms.len());
        for t1 in lhs_terms.iter() {
            for t2 in rhs_terms.iter() {
                result_terms.push(t1 * t2);
            }
        }
        Expression(result_terms)
    }
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

fn normal_order_many(terms: Expression) -> Expression {
    let mut result: Expression = Expression(Vec::new());
    for term in terms.0 {
        let mut normal_terms = normal_order(term);
        result.0.append(&mut normal_terms.0);
    }
    consolidate(result)
}

fn normal_order(term: Term) -> Expression {
    let mut queue = vec![term];
    let mut result: Expression = Expression(Vec::new());

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
        result.0.push(term);
    }

    consolidate(result)
}

fn consolidate(terms: Expression) -> Expression {
    let mut consolidated: HashMap<Vec<Operator>, f32> = HashMap::new();

    for term in terms.0 {
        *consolidated.entry(term.ops).or_insert(0.0) += term.coeff;
    }

    let mut result: Vec<Term> = consolidated
        .into_iter()
        .filter(|(_, c)| c.abs() >= 1e-6)
        .map(|(ops, coeff)| Term { coeff, ops })
        .collect();
    
    result.sort_by(|a, b| a.ops.len().cmp(&b.ops.len()));
    Expression(result)
}





fn main() {
    let input = Term {
        coeff: 1.0, 
        ops: vec![
            Operator::creation(1),
            Operator::annihilation(1),
            Operator::creation(1),
    ]};

    let result = normal_order(input.clone());
    println!("Input: {:?}", input);
    println!("Output:");
    for term in result.0 {
        println!("{:?}", term);
    }

    let input2 = Term {coeff: 1.0, 
        ops: vec![
        Operator::annihilation(1),
        Operator::annihilation(2),
        Operator::creation(2),
        Operator::creation(1),
    ]};

    println!("\nInput 2: {:?}", input2);
    let result2 = normal_order(input2);
    for term in result2.0 {
        println!("{:?}", term);
    }


    let input3 = Expression(vec![
        Term {coeff: 1.0, 
        ops: vec![
            Operator::annihilation(2),
            Operator::creation(1),
        ]},
        Term {coeff: 1.0, 
        ops: vec![
            Operator::creation(1),
            Operator::annihilation(2),
        ]},
    ]);

    println!("\nInput 3: {:?}", input3);
    let result3 = normal_order_many(input3);
    for term in result3.0 {
        println!("{:?}", term);
    }
    
    {
        let a1 = Term::density(1);
        let b1 = Term::density(2);
        let input4 = a1 * b1 * 42.0;
        println!("\nInput 4: {:?}", input4);
    }

    {
        let a2 = Term::density(2);
        let b2 = Term::density(2);
        let input5 = a2 + b2;
        println!("\nInput 5: {:?}", input5);
    }

    {
        let e1 = Term::density(1) + Term::density(2);
        let e2 = Term::density(2) + Term::density(3);
        let input6 = e1 * e2 * 3.0;
        println!("\nInput 6: {:?}", input6);
    }

    {
        let hamiltonian = Expression::hopping(1, 2) + 0.5 * Term::density(1) * Term::density(2);
        println!("\nHamiltonian: {:?}", normal_order_many(hamiltonian));
    }
}
