use std::collections::HashMap;
use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul};
use std::f64::consts::PI;
use num_complex::{Complex, ComplexFloat};
use faer::Mat;

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

    fn adjoint(&self) -> Operator {
        match self.op {
            OpType::Creation     => Operator::annihilation(self.index),
            OpType::Annihilation => Operator::creation(self.index),
        }
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
    coeff: Complex<f64>,
    ops: Vec<Operator>,
}

impl Term {
    fn adjoint(&self) -> Term {
        let mut new_ops = self.ops.clone();
        new_ops.reverse();
        for op in &mut new_ops {
            *op = op.adjoint();
        }
        Term {
            coeff: self.coeff.conj(),
            ops: new_ops,
        }
    }
    
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
        write!(f, "({:.6}, {:.6}) × {:?}", self.coeff.re, self.coeff.im, self.ops)
    }
}

#[derive(Clone, PartialEq)]
struct Expression(Vec<Term>);

impl Expression {
    fn new() -> Expression {
        Expression(Vec::new())
    }

    fn scalar(c: Complex<f64>) -> Expression {
        Expression(vec![Term { coeff: c, ops: vec![] }])
    }

    fn adjoint(&self) -> Expression {
        let mut new_terms = Vec::with_capacity(self.0.len());
        for term in &self.0 {
            new_terms.push(term.adjoint());
        }
        Expression(new_terms)
    }

    pub fn consolidate(&mut self) {
        if self.0.is_empty() { return; }

        let mut consolidated: HashMap<Vec<Operator>, Complex<f64>> = HashMap::new();
        
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


impl From<f64> for Term {
    fn from(c: f64) -> Self {
        Term { coeff: c.into(), ops: vec![] }
    }
}

impl From<Complex<f64>> for Term {
    fn from(c: Complex<f64>) -> Self {
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

impl From<f64> for Expression {
    fn from(c: f64) -> Self {
        Expression(vec![Term::from(c)])
    }
}

impl From<Complex<f64>> for Expression {
    fn from(c: Complex<f64>) -> Self {
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

impl Add<Expression> for f64 {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Self::Output {
        Expression::from(self) + rhs
    }
}

impl Add<Term> for f64 {
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

impl Sub<Expression> for f64 {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Self::Output {
        let mut rhs_expr = rhs;
        for term in &mut rhs_expr.0 {
            term.coeff = -term.coeff;
        }
        Expression::from(self) + rhs_expr
    }
}

impl Sub<Term> for f64 {
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

impl Mul<Expression> for f64 {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let mut result = rhs;
        for term in &mut result.0 {
            term.coeff *= self;
        }
        result
    }
}

impl Mul<Term> for f64 {
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

fn normal_order_many(terms: &Expression) -> Expression {
    let mut result = Expression::new();
    for term in &terms.0 {
        let mut normal_terms = normal_order(term);
        result.0.append(&mut normal_terms.0);
    }
    result.consolidate();
    result
}

fn normal_order(term: &Term) -> Expression {
    let mut queue = vec![term.clone()];
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


/// A basis of N particles and M sites can be represented as a vector of terms.
/// Each term represents a product of operators acting on the vacuum state.
/// E.g., for N=2 and M=3, a basis state could be represented as:
/// a+ (0) a+ (2) | 0 >
/// which corresponds to two particles at sites 0 and 2.
struct Basis {
    n_particles: usize,
    n_sites: usize,
    states: Vec<Term>,
}

/// Because of bosonic statistics, we know that the size of the basis is given by:
/// C(N + M - 1, N) = (N + M - 1)! / (N! * (M - 1)!)
/// where C(n, k) is the binomial coefficient "n choose k".
fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut res = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

/// Bosonic state normalization factor:
/// For a state with occupation numbers n1, n2, ..., nM at sites 1 to M,
/// the normalization factor is given by:
/// 1 / sqrt(n1! * n2! * ... * nM!)
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn bosonic_normalization(ops: &Vec<Operator>) -> f64 {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for op in ops {
        match op.op {
            OpType::Creation => {
                *counts.entry(op.index).or_insert(0) += 1;
            },
            OpType::Annihilation => panic!("Shouldn't have annihilation operator here.")
        }
    }
    let mut norm = 1.0;
    for &count in counts.values() {
        norm *= factorial(count) as f64;
    }
    1.0 / norm.sqrt()
}

/// Construct the basis of N particles and M sites.
fn construct_basis(n_particles: usize, n_sites: usize) -> Basis {
    let n_states = choose(n_particles + n_sites - 1, n_particles);
    let mut states = Vec::with_capacity(n_states);

    fn generate_states(n_particles: usize, n_sites: usize, start_site: usize, current_state: &mut Vec<Operator>, states: &mut Vec<Term>) {
        if n_particles == 0 {
            let normalization = bosonic_normalization(current_state);
            states.push(Term { coeff: normalization.into(), ops: current_state.clone() });
            return;
        }
        for site in start_site..n_sites {
            current_state.push(Operator::creation(site));
            generate_states(n_particles - 1, n_sites, site, current_state, states);
            current_state.pop();
        }
    }

    let mut current_state = Vec::new();
    generate_states(n_particles, n_sites, 0, &mut current_state, &mut states);

    Basis { n_particles, n_sites, states }
}


/// Computes matrix elements of an operator using a basis.
/// Naively, we can compute <bra|O|ket> for all bra and ket in the basis.
/// Using the normal ordering function, we can compute O|ket> and then take the inner product with <bra|,
/// the matrix element is then given by the coefficient of the vacuum state in the resulting expression.
fn compute_matrix_elements(basis: &Basis, operator: &Expression) -> Mat<Complex<f64>> {
    let n_states = basis.states.len();
    let mut matrix = Mat::zeros(n_states, n_states);
    for (i, bra) in basis.states.iter().enumerate() {
        for (j, ket) in basis.states.iter().enumerate() {
            let product = bra.adjoint() * operator.clone() * ket.clone();
            let normal_ordered = normal_order_many(&product);
            matrix[(i, j)] = normal_ordered.0.iter()
                .find(|term| term.ops.is_empty())
                .map_or(Complex::new(0.0, 0.0), |term| term.coeff);
        }
    }
    matrix
}



/// Perform Fourier transform on a single operator in 1D
fn fourier_transform_operator(op: &Operator, n_sites: usize) -> Expression {
    let mut terms = Vec::new();
    let I = Complex::new(0.0, 1.0);
    for k in 0..n_sites {
        let (sign, new_op) = match op.op {
            OpType::Creation     => (-1.0, Operator::creation(k)),
            OpType::Annihilation => ( 1.0, Operator::annihilation(k)),
        };
        let phase = (sign * 2.0 * PI * I * (op.index as f64) * (k as f64) / (n_sites as f64)).exp();
        terms.push(Term { coeff: phase / (n_sites as f64).sqrt(), ops: vec![new_op] });
    }
    Expression(terms)
}

fn fourier_transform_term(term: &Term, n_sites: usize) -> Expression {
    let mut result = Expression::scalar(1.0.into());
    for op in &term.ops {
        let ft_op = fourier_transform_operator(op, n_sites);
        result = result * ft_op;
    }
    for t in &mut result.0 {
        t.coeff *= term.coeff;
    }
    result
}

fn fourier_transform_expression(expr: &Expression, n_sites: usize) -> Expression {
    let mut result = Expression::new();
    for term in &expr.0 {
        let ft_term = fourier_transform_term(&term, n_sites);
        result = result + ft_term;
    }
    result
}





/// Building blocks for common Hamiltonians.
struct BuildingBlocks;

impl BuildingBlocks {    
    fn chemical_potential_1d(mu: f64, n_sites: usize) -> Expression {
        let mut expr = Expression::new();
        for i in 0..n_sites {
            let density_i = Term::density(i);
            expr = expr + density_i * (-mu);
        }
        expr
    }
    
    fn hopping_1d(t: f64, n_sites: usize) -> Expression {
        let mut expr = Expression::new();
        for i in 0..n_sites {
            let j = (i + 1) % n_sites;
            expr = expr + Expression::hopping(i, j) * (-t);
        }
        expr
    }

    fn hubbard_u_1d(u: f64, n_sites: usize) -> Expression {
        let mut expr = Expression::new();
        for i in 0..n_sites {
            let density_i = Term::density(i);
            expr = expr + (density_i.clone() * (density_i - 1.0)) * (u / 2.0);
        }
        expr
    }
}




/// Model trait.
/// Every model should provide a method to generate its Hamiltonian as an Expression.
trait Model {
    fn hamiltonian(&self) -> Expression;
}

struct Chain {
    t: f64,
    mu: f64,
    n_sites: usize,
}

impl Model for Chain {
    fn hamiltonian(&self) -> Expression {
        BuildingBlocks::hopping_1d(self.t, self.n_sites) +
        BuildingBlocks::chemical_potential_1d(self.mu, self.n_sites)
    }
}

struct BoseHubbard1D {
    t: f64,
    u: f64,
    mu: f64,
    n_sites: usize,
}

impl Model for BoseHubbard1D {
    fn hamiltonian(&self) -> Expression {
        BuildingBlocks::hopping_1d(self.t, self.n_sites) +
        BuildingBlocks::hubbard_u_1d(self.u, self.n_sites) +
        BuildingBlocks::chemical_potential_1d(self.mu, self.n_sites)
    }
}


fn main() {
    let input = Term {
        coeff: 1.0.into(), 
        ops: vec![
            Operator::creation(1),
            Operator::annihilation(1),
            Operator::creation(1),
    ]};

    let result = normal_order(&input);
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
    let result2 = normal_order(&input2);
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
    let result3 = normal_order_many(&input3);
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
        println!("\nHamiltonian: {:?}", normal_order_many(&hamiltonian));
    }

    {
        let hamiltonian = 
        Operator::creation(1) * (1.0 - Term::density(1)) * Operator::annihilation(2) +
        Operator::creation(2) * (1.0 - Term::density(2)) * Operator::annihilation(1);
        println!("\nHamiltonian 2: {:?}", normal_order_many(&hamiltonian));
    }

    {
        let op = Operator::creation(1);
        let ft_op = fourier_transform_operator(&op, 4);
        println!("\nFourier Transform of {:?}:", op);
        for term in ft_op.0 {
            println!("{:?}", term);
        }
    }

    {
        let term = Term::density(1);
        let ft_term = fourier_transform_term(&term, 4);
        println!("\nFourier Transform of density term:");
        for term in ft_term.0 {
            println!("{:?}", term);
        }
    }

    {
        let expr = Term::density(1) + Term::density(2);
        let ft_expr = fourier_transform_expression(&expr, 4);
        println!("\nFourier Transform of density expression:");
        for term in ft_expr.0 {
            println!("{:?}", term);
        }
    }

    {
        let expr = Expression::hopping(0, 1) + Expression::hopping(1, 2) + Expression::hopping(2, 0);
        let ft_expr = fourier_transform_expression(&expr, 3);
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
        let normal_hamiltonian = normal_order_many(&hamiltonian);
        let ft_hamiltonian = fourier_transform_expression(&normal_hamiltonian, 3);
        let ft_ordered_hamiltonian = normal_order_many(&ft_hamiltonian);
        println!("\nFourier Transform of full Hamiltonian:");
        for term in ft_ordered_hamiltonian.0 {
            println!("{:?}", term);
        }
    }

    {
        let basis = construct_basis(2, 3);
        println!("\nBasis states for {} particles in {} sites:", basis.n_particles, basis.n_sites);
        for (i, state) in basis.states.iter().enumerate() {
            println!("State {}: {:?}", i, state);
        }
    }

    {
        let basis = construct_basis(3, 2);
        println!("\nBasis states for {} particles in {} sites:", basis.n_particles, basis.n_sites);
        for (i, state) in basis.states.iter().enumerate() {
            println!("State {}: {:?}", i, state);
        }
    }

    {
        let basis = construct_basis(2, 4);
        println!("\nBasis states for {} particles in {} sites:", basis.n_particles, basis.n_sites);
        for (i, state) in basis.states.iter().enumerate() {
            println!("State {}: {:?}", i, state);
        }
    }

    {
        let basis = construct_basis(1, 4);
        println!("\nBasis states for {} particles in {} sites:", basis.n_particles, basis.n_sites);
        for (i, state) in basis.states.iter().enumerate() {
            println!("State {}: {:?}", i, state);
        }
        
        let model = Chain { t: 1.0, mu: 0.0, n_sites: basis.n_sites };
        let hamiltonian = model.hamiltonian();
        let matrix = compute_matrix_elements(&basis, &hamiltonian);
        println!("\nHamiltonian matrix:");
        println!("{:?}", matrix);
        
        let eig = matrix.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        println!("\nEigenvalues:");
        println!("{:?}", eig);
    }

    {
        let basis = construct_basis(2, 4);
        println!("\nBasis states for {} particles in {} sites:", basis.n_particles, basis.n_sites);
        for (i, state) in basis.states.iter().enumerate() {
            println!("State {}: {:?}", i, state);
        }
        
        let model = BoseHubbard1D { t: 1.0, u: 2.0, mu: 0.0, n_sites: basis.n_sites };
        let hamiltonian = model.hamiltonian();
        let matrix = compute_matrix_elements(&basis, &hamiltonian);
        println!("\nHamiltonian matrix:");
        println!("{:?}", matrix);
        
        let eig = matrix.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        println!("\nEigenvalues:");
        println!("{:?}", eig);
    }
}
