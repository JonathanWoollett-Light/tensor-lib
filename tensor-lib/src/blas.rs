fn trans(x: bool) -> cblas::Transpose {
    match x {
        true => cblas::Transpose::Ordinary,
        false => cblas::Transpose::None,
    }
}

/// Unit struct on which BLAS functionality is implemented. This allows for effective function overloading for the various permutations of vector/matrix/tensor shapes.
///
/// ### Support
/// #### Level 1
///
/// Single|Double|Complex|Double Complex
/// ---|---|---|---
/// SSWAP|DSWAP||
/// SSCAL|DSCAL||
/// SCOPY|DCOPY||
/// SAXPY|DAXPY||
/// SDOT|DDOT||
/// SNRM2|DNRM2||
/// SASUM|DASUM||
/// ISAMAX|IDAMAX||
///
/// #### Level 2
///
/// Single|Double|Complex|Double Complex
/// ---|---|---|---
///
/// #### Level 3
///
/// Single|Double|Complex|Double Complex
/// ---|---|---|---
/// SGEMM|DGEMM||
///
pub struct BLAS;

/// Defines tensors of generic shapes but specific types.
pub trait Tensor<T> {
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
}
// BLAS level 1
// ---------------------------------------------------------------------------
/// [SSWAP](http://www.netlib.org/lapack/explore-html/d9/da9/sswap_8f.html) BLAS operation.
pub trait SSWAP<X, Y> {
    fn sswap(x: &mut X, y: &mut Y);
}
/// [DSWAP](http://www.netlib.org/lapack/explore-html/db/dd4/dswap_8f.html) BLAS operation.
pub trait DSWAP<X, Y> {
    fn dswap(x: &mut X, y: &mut Y);
}
/// [SSCAL](http://www.netlib.org/lapack/explore-html/d9/d04/sscal_8f.html) BLAS operation.
pub trait SSCAL<X> {
    fn sscal(alpha: f32, x: &mut X);
}
/// [DSCAL](http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga793bdd0739bbd0e0ec8655a0df08981a.html#ga793bdd0739bbd0e0ec8655a0df08981a) BLAS operation.
pub trait DSCAL<X> {
    fn dscal(alpha: f64, x: &mut X);
}
/// [SCOPY](http://www.netlib.org/lapack/explore-html/de/dc0/scopy_8f.html) BLAS operation.
pub trait SCOPY<X, Y> {
    fn scopy(x: &X, y: &mut Y);
}
/// [DCOPY](http://www.netlib.org/lapack/explore-html/alpha/d6c/dcopy_8f.html) BLAS operation.
pub trait DCOPY<X, Y> {
    fn dcopy(x: &X, y: &mut Y);
}
/// [SAXPY](http://www.netlib.org/lapack/explore-html/d8/daf/saxpy_8f.html) BLAS operation.
pub trait SAXPY<X, Y> {
    fn saxpy(alpha: f32, x: &X, y: &mut Y);
}
/// [DAXPY](http://www.netlib.org/lapack/explore-html/d9/dcd/daxpy_8f.html) BLAS operation.
pub trait DAXPY<X, Y> {
    fn daxpy(alpha: f64, x: &X, y: &mut Y);
}
/// [SDOT](http://www.netlib.org/lapack/explore-html/d0/d16/sdot_8f.html) BLAS operation.
pub trait SDOT<X, Y> {
    fn sdot(x: &X, y: &Y) -> f32;
}
/// [DDOT](http://www.netlib.org/lapack/explore-html/d5/df6/ddot_8f.html) BLAS operation.
pub trait DDOT<X, Y> {
    fn ddot(x: &X, y: &Y) -> f64;
}
/// [SASUM](http://www.netlib.org/lapack/explore-html/df/d1f/sasum_8f.html) BLAS operation.
pub trait SASUM<X> {
    fn sasum(x: &X) -> f32;
}
/// [DASUM](http://www.netlib.org/lapack/explore-html/de/d05/dasum_8f.html) BLAS operation.
pub trait DASUM<X> {
    fn dasum(x: &X) -> f64;
}
/// [SNRM2](https://www.netlib.org/lapack/explore-html/df/d28/group__single__blas__level1_gad179c1611098b5881f147d39afb009b8.html) BLAS operation.
pub trait SNRM2<X> {
    fn snrm2(x: &X) -> f32;
}
/// [DNRM2](http://www.netlib.org/lapack/explore-html/df/d28/group__single__blas__level1_gab5393665c8f0e7d5de9bd1dd2ff0d9d0.html) BLAS operation.
pub trait DNRM2<X> {
    fn dnrm2(x: &X) -> f64;
}
/// [ISAMAX](http://www.netlib.org/lapack/explore-html/d6/d44/isamax_8f.html) BLAS operation.
pub trait ISAMAX<X> {
    fn isamax(x: &X) -> usize;
}
/// [IDAMAX](http://www.netlib.org/lapack/explore-html/dd/de0/idamax_8f.html) BLAS operation.
pub trait IDAMAX<X> {
    fn idamax(x: &X) -> usize;
}
/// Internal wrapper around `cblas::sswap`.
pub(crate) fn sswap(x: &mut [f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    unsafe { cblas::sswap(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::dswap`.
pub(crate) fn dswap(x: &mut [f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    unsafe { cblas::dswap(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::sscal`
pub(crate) fn sscal(alpha: f32, x: &mut [f32]) {
    unsafe { cblas::sscal(x.len() as i32, alpha, x, 1) }
}
/// Internal wrapper around `cblas::dscal`.
pub(crate) fn dscal(alpha: f64, x: &mut [f64]) {
    unsafe { cblas::dscal(x.len() as i32, alpha, x, 1) }
}
/// Internal wrapper around `cblas::scopy`.
pub(crate) fn scopy(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    unsafe { cblas::scopy(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::dcopy`.
pub(crate) fn dcopy(x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    unsafe { cblas::dcopy(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::saxpy`
pub(crate) fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    unsafe { cblas::saxpy(x.len() as i32, alpha, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::daxpy`.
pub(crate) fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    unsafe { cblas::daxpy(x.len() as i32, alpha, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::saxpy`
pub(crate) fn sdot(x: &[f32], y: &[f32]) -> f32 {
    unsafe { cblas::sdot(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::daxpy`.
pub(crate) fn ddot(x: &[f64], y: &[f64]) -> f64 {
    unsafe { cblas::ddot(x.len() as i32, x, 1, y, 1) }
}
/// Internal wrapper around `cblas::sasum`
pub(crate) fn sasum(x: &[f32]) -> f32 {
    unsafe { cblas::sasum(x.len() as i32, x, 1) }
}
/// Internal wrapper around `cblas::dasum`.
pub(crate) fn dasum(x: &[f64]) -> f64 {
    unsafe { cblas::dasum(x.len() as i32, x, 1) }
}
/// Internal wrapper for `cblas::snrm2`.
pub(crate) fn snrm2(x: &[f32]) -> f32 {
    unsafe { cblas::snrm2(x.len() as i32, x, 1) }
}
/// Internal wrapper for `cblas::dnrm2`.
pub(crate) fn dnrm2(x: &[f64]) -> f64 {
    unsafe { cblas::dnrm2(x.len() as i32, x, 1) }
}
/// Internal wrapper for `cblas::isamax`.
pub(crate) fn isamax(x: &[f32]) -> usize {
    unsafe { cblas::isamax(x.len() as i32, x, 1) as usize }
}
/// Internal wrapper for `cblas::idamax`.
pub(crate) fn idamax(x: &[f64]) -> usize {
    unsafe { cblas::idamax(x.len() as i32, x, 1) as usize }
}
// It doesn't matter the shape of a tensor as long as its type is f32 or f64 we can do a variety of
//  BLAS operations.
// --------------------------------------
impl<X: Tensor<f32>> SSCAL<X> for BLAS {
    fn sscal(alpha: f32, x: &mut X) {
        sscal(alpha, x.data_mut());
    }
}
impl<X: Tensor<f64>> DSCAL<X> for BLAS {
    fn dscal(alpha: f64, x: &mut X) {
        dscal(alpha, x.data_mut());
    }
}
impl<X: Tensor<f32>> SASUM<X> for BLAS {
    fn sasum(x: &X) -> f32 {
        sasum(x.data())
    }
}
impl<X: Tensor<f64>> DASUM<X> for BLAS {
    fn dasum(x: &X) -> f64 {
        dasum(x.data())
    }
}
impl<X: Tensor<f32>> SNRM2<X> for BLAS {
    fn snrm2(x: &X) -> f32 {
        snrm2(x.data())
    }
}
impl<X: Tensor<f64>> DNRM2<X> for BLAS {
    fn dnrm2(x: &X) -> f64 {
        dnrm2(x.data())
    }
}
impl<X: Tensor<f32>> ISAMAX<X> for BLAS {
    fn isamax(x: &X) -> usize {
        isamax(x.data())
    }
}
impl<X: Tensor<f64>> IDAMAX<X> for BLAS {
    fn idamax(x: &X) -> usize {
        idamax(x.data())
    }
}
// BLAS level 3
// ---------------------------------------------------------------------------
/// [DGEMM](http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html) BLAS operation.
#[cfg(feature = "d2")]
pub trait DGEMM<A, B, C> {
    /// c = alpha * a@b + beta*c
    fn dgemm(a: &A, b: &B, c: &mut C, alpha: f64, beta: f64);
}
/// [SGEMM](http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html) BLAS operation.
#[cfg(feature = "d2")]
pub trait SGEMM<A, B, C> {
    /// c = alpha * a@b + beta*c
    fn sgemm(a: &A, b: &B, c: &mut C, alpha: f32, beta: f32);
}
/// Internal wrapper around `cblas::sgemm`.
#[cfg(feature = "d2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(
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
/// Internal wrapper around `cblas::dgemm`.
#[cfg(feature = "d2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dgemm(
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
