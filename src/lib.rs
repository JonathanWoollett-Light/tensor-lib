#![allow(incomplete_features)]
#![feature(doc_cfg)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
//! ## Dimensionality features
//! To avoid unnecessarily including a huge quantity of code there are features for each step of
//!  dimensionality. If you only need vectors and matrices then `d2` will provide these, if you 
//!  need vectors matrices and 3d tensors then `d3` will provide this (`d3` is the current 
//!  `default` feature), etc.
//!
//! It is important to note you only need to specify the feature for the highest dimensionality you
//!  require. Enabling `d5` and `d3` is no different than only enabling `d5`.
//! 
//! Current support goes up to `d10`. I haven't tried anything past `d6`, the higher 
//!  dimensionalities generate exponentially more code, it is quite possible `d10` generates many 
//!  gigabytes of code and possibly simply breaks. I would highly recommended using `default` or 
//!  the minimum dimensionality you can, otherwise you will find compile times become particularly 
//!  laborious.
//! ## Examples
//! ### Construction
//! ```
//! use tensor_lib::*;
//! // Vectors
//! let a = VectorD::<i32>::from((2,vec![1,2]));
//! let b = VectorS::<i32,2>::from((vec![1,2]));
//! // ┌─────┐
//! // │ 1 2 │
//! // └─────┘
//! // Matrices
//! let c = MatrixDxD::<i32>::from((2,3,vec![1,2,3,4,5,6]));
//! let d = MatrixDxS::<i32,2>::from((3,vec![1,2,3,4,5,6]));
//! let e = MatrixSxD::<i32,2>::from((3,vec![1,2,3,4,5,6]));
//! let f = MatrixSxS::<i32,2,3>::from((vec![1,2,3,4,5,6]));
//! // ┌───────┐
//! // │ 1 2 3 │
//! // │ 4 5 6 │
//! // └───────┘
//! // Tensors
//! let g = Tensor3DxDxD::<i32>::from((2,3,2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let h = Tensor3DxDxS::<i32,2>::from((2,3,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let i = Tensor3DxSxD::<i32,3>::from((2,2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let j = Tensor3SxDxD::<i32,2>::from((3,2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let k = Tensor3DxSxS::<i32,3,2>::from((2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let l = Tensor3SxSxD::<i32,2,3>::from((2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! let m = Tensor3SxSxS::<i32,2,3,2>::from((vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! // ┌──────────┐
//! // │  1  2  3 │
//! // │  4  5  6 │   
//! // └──────────┘
//! // ┌──────────┐
//! // │  7  8  9 │
//! // │ 10 11 12 │
//! // └──────────┘
//! ```
//! ```
//! use rand::distributions::{Uniform, Standard};
//! use tensor_lib::*;
//! let a = VectorD::<i32>::from_distribution((2,Uniform::<i32>::from(0..10)));
//! let b = MatrixDxS::<f32,2>::from_distribution((3,Standard));
//! let c = Tensor3DxSxD::<i32,3>::from_distribution((2,2,Uniform::<i32>::from(5..15)));
//! ```
//! ### Indexing
//! ```
//! use tensor_lib::*;
//! // 2
//! let a = VectorS::<i32,2>::from((vec![1,2]));
//! assert_eq!(1,a[[0]]);
//! assert_eq!(2,a[[1]]);
//! // 3x2
//! let b = MatrixDxS::<i32,3>::from((2,vec![1,2,3,4,5,6]));
//! assert_eq!(1,b[[0,0]]);
//! assert_eq!(2,b[[1,0]]);
//! assert_eq!(3,b[[0,1]]);
//! assert_eq!(4,b[[1,1]]);
//! assert_eq!(5,b[[0,2]]);
//! assert_eq!(6,b[[1,2]]);
//! // 2x3x2
//! let c = Tensor3DxSxS::<i32,3,2>::from((2,vec![1,2,3,4,5,6,7,8,9,10,11,12]));
//! assert_eq!(1,c[[0,0,0]]);
//! assert_eq!(2,c[[1,0,0]]);
//! assert_eq!(3,c[[0,1,0]]);
//! assert_eq!(4,c[[1,1,0]]);
//! assert_eq!(5,c[[0,2,0]]);
//! assert_eq!(6,c[[1,2,0]]);
//! assert_eq!(7,c[[0,0,1]]);
//! assert_eq!(8,c[[1,0,1]]);
//! assert_eq!(9,c[[0,1,1]]);
//! assert_eq!(10,c[[1,1,1]]);
//! assert_eq!(11,c[[0,2,1]]);
//! assert_eq!(12,c[[1,2,1]]);
//! ```
//! ### Arithmetic
//! Most of [`std::ops`] are all implemented as their respective component-wise implementations.
//! ```
//! use tensor_lib::*;
//! let a = VectorD::<i32>::from((2,vec![1,2]));
//! let b = VectorS::<i32,2>::from((vec![7,3]));
//! let c = a + b;
//! assert_eq!(c, VectorS::<i32,2>::from((vec![8,5])));
//! let d = MatrixDxD::<i32>::from((2,3,vec![1,2,3,4,5,6]));
//! let e = MatrixSxD::<i32,2>::from((3,vec![7,8,9,10,11,12]));
//! let f = d + e;
//! assert_eq!(f, MatrixSxD::<i32,2>::from((3,vec![8,10,12,14,16,18])));
//! ```
//! ### Slicing
//! **When slicing you need to include `#![feature(adt_const_params)]` and `#![feature(generic_const_exprs)]` otherwise you will get UB.**
//! ```
//! #![allow(incomplete_features)]
//! #![feature(adt_const_params)]
//! #![feature(generic_const_exprs)]
//! use tensor_lib::*;
//! let a = VectorD::<i32>::from((2,vec![1,2]));
//! assert_eq!(Slice1S::slice::<{0..2}>(&a), VectorS::<&i32,2>::from((vec![&1,&2])));
//! let b = MatrixDxD::<i32>::from((2,3,vec![1,2,3,4,5,6]));
//! assert_eq!(Slice2SxD::slice::<{0..1}>(&b,0..3), MatrixSxD::<&i32,1>::from((3,vec![&1,&2,&3])));
//! ```
//! ### BLAS
//! ```ignore
//! use tensor_lib::*;
//!
//! const M: usize = 2;
//! const N: usize = 2;
//! const K: usize = 2;
//!
//! let a = MatrixSxS::<f64, M, K>::from(vec![1., 2., 3., 4.]);
//! let b = MatrixSxS::<f64, K, N>::from(vec![5., 6., 7., 8.]);
//! let mut c = MatrixSxS::<f64, M, N>::from(vec![0., 0., 0., 0.]);
//!
//! BLAS::dgemm(&a, &b, &mut c, 1., 1.);
//! assert_eq!(c.data, vec![19., 22., 43., 50.])
//! ```
//! ### Display
//! All tensors implement [`std::fmt::Display`], like:
//! ```text
//! // VectorS::<i32,7>::from((vec![1, 20, 300, 40, 5, 60, 7]));
//! ┌                    ┐
//! │ 1 20 300 40 5 60 7 │
//! └                    ┘
//! // MatrixSxS::<i32,4,4>::from((vec![10, 2, 3, 4, 5, 6, 700, 8, 90, 10, 110, 12, 130, 140, 150, 160,]));
//! ┌                 ┐
//! │  10   2   3   4 │
//! │   5   6 700   8 │
//! │  90  10 110  12 │
//! │ 130 140 150 160 │
//! └                 ┘
//! // Tensor3SxSxS::<i32,3,3,3>::from((vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27,]));
//! ┌                                    ┐
//! │┌          ┐┌          ┐┌          ┐│
//! ││  1  2  3 ││ 10 11 12 ││ 19 20 21 ││
//! ││  4  5  6 ││ 13 14 15 ││ 22 23 24 ││
//! ││  7  8  9 ││ 16 17 18 ││ 25 26 27 ││
//! │└          ┘└          ┘└          ┘│
//! └                                    ┘
//! // Tensor4SxSxSxS::<i32,2,3,2,2>::from((vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,]));
//! ┌                  ┐
//! │┌       ┐┌       ┐│
//! ││  1  2 ││  7  8 ││
//! ││  3  4 ││  9 10 ││
//! ││  5  6 ││ 11 12 ││
//! │└       ┘└       ┘│
//! │┌       ┐┌       ┐│
//! ││ 13 14 ││ 19 20 ││
//! ││ 15 16 ││ 21 22 ││
//! ││ 17 18 ││ 23 24 ││
//! │└       ┘└       ┘│
//! └                  ┘
//! ```
//! The format follows:
//! ```text
//! ...
//! . ┌──5
//! . │ ┌──3
//!   6 │ ┌──1
//!     4 │ 
//!       2 
//! ```
//! The number corresponding to the dimensions, where for example in a `Tensor6SxSxSxSxSxS`, `A->1`, `B->2`, `C->3`, `D->4`, `E->5`, `F->6`.
extern crate openblas_src;
use tensor_lib_macros::tensors;

/// Internal function for getting the length of a `usize` range.
pub const fn range_len(x: std::ops::Range<usize>) -> usize {
    x.end - x.start
}
// pub trait Slice1D {
//     fn slice<T>(x: std::ops::Range<usize>) -> Tensor1D<T>;
// }
// pub trait Slice1S {
//     fn slice<T, const X: std::ops::Range<usize>>() -> Tensor1S<T, { range_len(X) }>;
// }

#[cfg(feature = "d10")]
tensors!(10);
#[cfg(all(feature = "d9", not(feature = "d10")))]
tensors!(9);
#[cfg(all(feature = "d8", not(feature = "d9")))]
tensors!(8);
#[cfg(all(feature = "d7", not(feature = "d8")))]
tensors!(7);
#[cfg(all(feature = "d6", not(feature = "d7")))]
tensors!(6);
#[cfg(all(feature = "d5", not(feature = "d6")))]
tensors!(5);
#[cfg(all(feature = "d5", not(feature = "d6")))]
tensors!(5);
#[cfg(all(feature = "d4", not(feature = "d5")))]
tensors!(4);
#[cfg(all(feature = "d3", not(feature = "d4")))]
tensors!(3);
#[cfg(all(feature = "d2", not(feature = "d3")))]
tensors!(2);

/// Unit struct on which BLAS functionality is implemented. This allows for effective function overloading for the various permutations of vector/matrix/tensor shapes.
pub struct BLAS();
/// [DGEMM](http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html) BLAS operation.
pub trait DGEMM<A, B, C> {
    /// c = alpha * a@b + beta*c
    fn dgemm(a: &A, b: &B, c: &mut C, alpha: f64, beta: f64);
}
/// [SGEMM](http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html) BLAS operation.
pub trait SGEMM<A, B, C> {
    /// c = alpha * a@b + beta*c
    fn sgemm(a: &A, b: &B, c: &mut C, alpha: f32, beta: f32);
}

/// Internal function used for getting lengths of display strings for tensors.
///
/// We allow dead code since this *likely* isn't dead code it is used by implementations produced by our macro.
#[allow(dead_code)]
fn display_bounds(h_dims: &[usize], max_str_width: usize) -> (Vec<String>, Vec<String>) {
    let lens = display_bound_lengths(h_dims, max_str_width);
    return lens
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, len)| {
            let border = "│".repeat(i);
            let num = if i > 0 { h_dims[i - 1] } else { 1 };
            let space = " ".repeat(len);
            let up = format!("┌{}┐", space).repeat(num);
            let down = format!("└{}┘", space).repeat(num);
            (
                format!("{}{}{}", border, up, border),
                format!("{}{}{}", border, down, border),
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    fn display_bound_lengths(h_dims: &[usize], max_str_width: usize) -> Vec<usize> {
        if h_dims.len() == 1 {
            vec![h_dims[0] * (max_str_width + 1) + 1]
        } else {
            let mut temp = display_bound_lengths(&h_dims[1..], max_str_width);
            let new_val = (temp[0] + 2) * h_dims[0];
            temp.push(new_val);
            temp
        }
    }
}

fn trans(x: bool) -> cblas::Transpose {
    match x {
        true => cblas::Transpose::Ordinary,
        false => cblas::Transpose::None,
    }
}
/// Internal convenience wrapper around `cblas::sgemm`.
fn sgemm(
    // If `true` then `op(a)=transpose(a)` else if `false` `op(a)=a`.
    trans_a: bool,
    // If `true` then `op(b)=transpose(b)` else if `false` `op(b)=b`.
    trans_b: bool,
    // Rows of op(a), rows of `c`.
    m: usize,
    // Columns of op(b), columns of `c`.
    n: usize,
    // Columns of op(a), rows of op(b).
    k: usize,
    alpha: f32,
    beta: f32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k, "Wrong shape of a");
    assert_eq!(b.len(), n * k, "Wrong shape of b");
    assert_eq!(c.len(), m * n, "Wrong shape of c");
    let (m, n, k) = (m as i32, n as i32, k as i32);
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            trans(trans_a),
            trans(trans_b),
            m,
            n,
            k,
            alpha,
            a,
            k,
            b,
            n,
            beta,
            c,
            n,
        );
    }
}
/// Internal convenience wrapper around `cblas::dgemm`.
fn dgemm(
    // If `true` then `op(a)=transpose(a)` else if `false` `op(a)=a`.
    trans_a: bool,
    // If `true` then `op(b)=transpose(b)` else if `false` `op(b)=b`.
    trans_b: bool,
    // Rows of op(a), rows of `c`.
    m: usize,
    // Columns of op(b), columns of `c`.
    n: usize,
    // Columns of op(a), rows of op(b).
    k: usize,
    alpha: f64,
    beta: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    assert_eq!(a.len(), m * k, "Wrong shape of a");
    assert_eq!(b.len(), n * k, "Wrong shape of b");
    assert_eq!(c.len(), m * n, "Wrong shape of c");
    let (m, n, k) = (m as i32, n as i32, k as i32);
    unsafe {
        cblas::dgemm(
            cblas::Layout::RowMajor,
            trans(trans_a),
            trans(trans_b),
            m,
            n,
            k,
            alpha,
            a,
            k,
            b,
            n,
            beta,
            c,
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::DGEMM;
    use crate::*;
    #[test]
    fn display() {
        let a = VectorS::<i32, 7>::from(vec![1, 20, 300, 40, 5, 60, 7]);
        // println!("a: {}", a);
        assert_eq!(a.to_string(),String::from("\n┌                             ┐\n│   1  20 300  40   5  60   7 │\n└                             ┘"));
        let b = MatrixSxS::<i32, 4, 4>::from(vec![
            10, 2, 3, 4, 5, 6, 700, 8, 90, 10, 110, 12, 130, 140, 150, 160,
        ]);
        // println!("b: {}", b);
        assert_eq!(b.to_string(),String::from("\n┌                 ┐\n│  10   2   3   4 │\n│   5   6 700   8 │\n│  90  10 110  12 │\n│ 130 140 150 160 │\n└                 ┘"));
        #[cfg(feature = "d3")]
        {
            let c = Tensor3SxSxS::<i32, 3, 3, 3>::from(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27,
            ]);
            // println!("c: {}", c);
            assert_eq!(c.to_string(),String::from("\n┌                                    ┐\n│┌          ┐┌          ┐┌          ┐│\n││  1  2  3 ││ 10 11 12 ││ 19 20 21 ││\n││  4  5  6 ││ 13 14 15 ││ 22 23 24 ││\n││  7  8  9 ││ 16 17 18 ││ 25 26 27 ││\n│└          ┘└          ┘└          ┘│\n└                                    ┘"));
        }
        #[cfg(feature = "d4")]
        {
            let d = Tensor4SxSxSxS::<i32, 2, 3, 2, 2>::from(vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24,
            ]);
            // println!("d: {}", d);
            assert_eq!(d.to_string(),String::from("\n┌                  ┐\n│┌       ┐┌       ┐│\n││  1  2 ││  7  8 ││\n││  3  4 ││  9 10 ││\n││  5  6 ││ 11 12 ││\n│└       ┘└       ┘│\n│┌       ┐┌       ┐│\n││ 13 14 ││ 19 20 ││\n││ 15 16 ││ 21 22 ││\n││ 17 18 ││ 23 24 ││\n│└       ┘└       ┘│\n└                  ┘"))
        }
    }
    #[test]
    fn dgemm() {
        const M: usize = 2;
        const N: usize = 2;
        const K: usize = 2;
        let a = crate::MatrixSxS::<f64, M, K>::from(vec![1., 2., 3., 4.]);
        let b = crate::MatrixSxS::<f64, K, N>::from(vec![5., 6., 7., 8.]);
        let mut c = crate::MatrixSxS::<f64, M, N>::from(vec![0., 0., 0., 0.]);
        println!("a: {:?}", a);
        println!("b: {:?}", b);
        println!("c: {:?}", c);

        crate::BLAS::dgemm(&a, &b, &mut c, 1., 1.);
        println!("a: {:?}", a);
        println!("b: {:?}", b);
        println!("c: {:?}", c);
        assert_eq!(c.data, vec![19., 22., 43., 50.])
    }
}
