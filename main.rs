use std::collections::HashMap;
use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul};
use std::f32::consts::PI;
use num_complex::{Complex, ComplexFloat};

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
    coeff: Complex<f32>,
    ops: Vec<Operator>,
}

impl Term {
    fn one_body(i: usize, j: usize) -> Term {
        Term {
            coeff: 1.0.into(),
            ops: vec![
                Operator::creation(i),
                Operator::annihilation(j),
            ],
        }
    }

    fn two_body(i: usize, j: usize, k: usize, l: usize) -> Term {
        Term {
            coeff: 1.0.into(),
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
        if (self.coeff.im - 0.0).abs() < 1e-6 {
            write!(f, "{} × {:?}", self.coeff.re, self.ops)
        } else {
            write!(f, "{} × {:?}", self.coeff, self.ops)
        }
    }
}

#[derive(Clone, PartialEq)]
struct Expression(Vec<Term>);

impl Expression {
    fn new() -> Expression {
        Expression(Vec::new())
    }

    fn scalar(c: Complex<f32>) -> Expression {
        Expression(vec![Term { coeff: c, ops: vec![] }])
    }

    pub fn consolidate(&mut self) {
        if self.0.is_empty() { return; }

        let mut consolidated: HashMap<Vec<Operator>, Complex<f32>> = HashMap::new();
        
        for term in self.0.drain(..) {
            *consolidated.entry(term.ops).or_insert(0.0.into()) += term.coeff;
        }

        self.0 = consolidated
            .into_iter()
            .filter(|(_, coeff)| coeff.abs() > 1e-6)
            .map(|(ops, coeff)| Term { coeff, ops })
            .collect();
        
        self.0.sort_by(|a, b| a.ops.len().cmp(&b.ops.len()));
    }
    
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


impl From<f32> for Term {
    fn from(c: f32) -> Self {
        Term { coeff: c.into(), ops: vec![] }
    }
}

impl From<Complex<f32>> for Term {
    fn from(c: Complex<f32>) -> Self {
        Term { coeff: c, ops: vec![] }
    }
}

impl From<Operator> for Term {
    fn from(op: Operator) -> Self {
        Term { coeff: 1.0.into(), ops: vec![op] }
    }
}

impl From<Term> for Expression {
    fn from(term: Term) -> Self {
        Expression(vec![term])
    }
}

impl From<Operator> for Expression {
    fn from(op: Operator) -> Self {
        Expression(vec![Term::from(op)])
    }
}

impl From<f32> for Expression {
    fn from(c: f32) -> Self {
        Expression(vec![Term::from(c)])
    }
}

impl From<Complex<f32>> for Expression {
    fn from(c: Complex<f32>) -> Self {
        Expression(vec![Term::from(c)])
    }
}




impl<T: Into<Expression>> Add<T> for Expression {
    type Output = Expression;

    fn add(mut self, rhs: T) -> Self::Output {
        self.0.extend(rhs.into().0);
        self.consolidate();
        self
    }
}

impl<T: Into<Expression>> Add<T> for Term {
    type Output = Expression;

    fn add(self, rhs: T) -> Self::Output {
        Expression::from(self) + rhs
    }
}

impl<T: Into<Expression>> Add<T> for Operator {
    type Output = Expression;

    fn add(self, rhs: T) -> Self::Output {
        Expression::from(self) + rhs
    }
}

impl Add<Expression> for f32 {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Self::Output {
        Expression::from(self) + rhs
    }
}

impl Add<Term> for f32 {
    type Output = Expression;

    fn add(self, rhs: Term) -> Self::Output {
        Expression::from(self) + rhs
    }
}



impl<T: Into<Expression>> Sub<T> for Expression {
    type Output = Expression;

    fn sub(mut self, rhs: T) -> Self::Output {
        let mut rhs_expr = rhs.into();
        for term in &mut rhs_expr.0 {
            term.coeff = -term.coeff;
        }
        self.0.extend(rhs_expr.0);
        self.consolidate();
        self
    }
}

impl<T: Into<Expression>> Sub<T> for Term {
    type Output = Expression;

    fn sub(self, rhs: T) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl<T: Into<Expression>> Sub<T> for Operator {
    type Output = Expression;

    fn sub(self, rhs: T) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Sub<Expression> for f32 {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Self::Output {
        let mut rhs_expr = rhs;
        for term in &mut rhs_expr.0 {
            term.coeff = -term.coeff;
        }
        Expression::from(self) + rhs_expr
    }
}

impl Sub<Term> for f32 {
    type Output = Expression;

    fn sub(self, rhs: Term) -> Self::Output {
        let mut rhs_expr = Expression::from(rhs);
        for term in &mut rhs_expr.0 {
            term.coeff = -term.coeff;
        }
        Expression::from(self) + rhs_expr
    }
}




impl<T: Into<Expression>> Mul<T> for Expression {
    type Output = Expression;
    
    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        let mut result_terms = Vec::new();
        
        for t1 in &self.0 {
            for t2 in &rhs_expr.0 {
                result_terms.push(t1 * t2);
            }
        }
        
        let mut result_expr = Expression(result_terms);
        result_expr.consolidate();
        result_expr
    }
}

impl<T: Into<Expression>> Mul<T> for Term {
    type Output = Expression;

    fn mul(self, rhs: T) -> Self::Output {
        Expression::from(self) * rhs
    }
}

impl<T: Into<Expression>> Mul<T> for Operator {
    type Output = Expression;

    fn mul(self, rhs: T) -> Self::Output {
        Expression::from(self) * rhs
    }
}

impl<'a, 'b> Mul<&'b Term> for &'a Term {
    type Output = Term;

    fn mul(self, rhs: &'b Term) -> Self::Output {
        let mut ops = Vec::with_capacity(self.ops.len() + rhs.ops.len());
        ops.extend_from_slice(&self.ops);
        ops.extend_from_slice(&rhs.ops);
        Term {
            coeff: self.coeff * rhs.coeff,
            ops,
        }
    }
}

impl Mul<Expression> for f32 {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let mut result = rhs;
        for term in &mut result.0 {
            term.coeff *= self;
        }
        result
    }
}

impl Mul<Term> for f32 {
    type Output = Term;
 
    fn mul(self, rhs: Term) -> Self::Output {
        let mut result = rhs;
        result.coeff *= self;
        result
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
    let mut result = Expression::new();
    for term in terms.0 {
        let mut normal_terms = normal_order(term);
        result.0.append(&mut normal_terms.0);
    }
    result.consolidate();
    result
}

fn normal_order(term: Term) -> Expression {
    let mut queue = vec![term];
    let mut result = Expression::new();

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
    result.consolidate();
    result
}


/// Perform Fourier transform on a single operator in 1D
fn fourier_transform_operator(op: Operator, n_sites: usize) -> Expression {
    let mut terms = Vec::new();
    let I = Complex::new(0.0, 1.0);
    for k in 0..n_sites {
        let phase = (2.0 * PI * I * (op.index as f32) * (k as f32) / (n_sites as f32)).exp();
        let new_op = match op.op {
            OpType::Creation     => Operator::creation(k),
            OpType::Annihilation => Operator::annihilation(k),
        };
        terms.push(Term { coeff: phase / (n_sites as f32).sqrt(), ops: vec![new_op] });
    }
    Expression(terms)
}

fn fourier_transform_term(term: Term, n_sites: usize) -> Expression {
    let mut result = Expression::scalar(1.0.into());
    for op in term.ops {
        let ft_op = fourier_transform_operator(op, n_sites);
        result = result * ft_op;
    }
    for t in &mut result.0 {
        t.coeff *= term.coeff;
    }
    result
}

fn fourier_transform_expression(expr: Expression, n_sites: usize) -> Expression {
    let mut result = Expression::new();
    for term in expr.0 {
        let ft_term = fourier_transform_term(term, n_sites);
        result = result + ft_term;
    }
    result
}


fn main() {
    let input = Term {
        coeff: 1.0.into(), 
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

    let input2 = Term {coeff: 1.0.into(), 
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
        Term {coeff: 1.0.into(), 
        ops: vec![
            Operator::annihilation(2),
            Operator::creation(1),
        ]},
        Term {coeff: 1.0.into(), 
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

    {
        let hamiltonian = 
        Operator::creation(1) * (1.0 - Term::density(1)) * Operator::annihilation(2) +
        Operator::creation(2) * (1.0 - Term::density(2)) * Operator::annihilation(1);
        println!("\nHamiltonian 2: {:?}", normal_order_many(hamiltonian));
    }

    {
        let op = Operator::creation(1);
        let ft_op = fourier_transform_operator(op, 4);
        println!("\nFourier Transform of {:?}:", op);
        for term in ft_op.0 {
            println!("{:?}", term);
        }
    }

    {
        let term = Term::density(1);
        let ft_term = fourier_transform_term(term, 4);
        println!("\nFourier Transform of density term:");
        for term in ft_term.0 {
            println!("{:?}", term);
        }
    }

    {
        let expr = Term::density(1) + Term::density(2);
        let ft_expr = fourier_transform_expression(expr, 4);
        println!("\nFourier Transform of density expression:");
        for term in ft_expr.0 {
            println!("{:?}", term);
        }
    }

    {
        let expr = Expression::hopping(0, 1) + Expression::hopping(1, 2) + Expression::hopping(2, 0);
        let ft_expr = fourier_transform_expression(expr, 3);
        println!("\nFourier Transform of hopping expression (should be diagonal):");
        for term in ft_expr.0 {
            println!("{:?}", term);
        }
    }

    {
        let hamiltonian = 
            Operator::creation(0) * (1.0 - Term::density(0)) * Operator::annihilation(1) + // hop from 0 to 1
            Operator::creation(1) * (1.0 - Term::density(1)) * Operator::annihilation(0) +
            Operator::creation(1) * (1.0 - Term::density(1)) * Operator::annihilation(2) + // hop from 1 to 2
            Operator::creation(2) * (1.0 - Term::density(2)) * Operator::annihilation(1) +
            Operator::creation(2) * (1.0 - Term::density(2)) * Operator::annihilation(0) + // hop from 2 to 0
            Operator::creation(0) * (1.0 - Term::density(0)) * Operator::annihilation(2);
        let normal_hamiltonian = normal_order_many(hamiltonian);
        let ft_hamiltonian = fourier_transform_expression(normal_hamiltonian, 3);
        println!("\nFourier Transform of full Hamiltonian:");
        for term in ft_hamiltonian.0 {
            println!("{:?}", term);
        }
    }
}
