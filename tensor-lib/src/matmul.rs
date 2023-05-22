extern crate openblas_src;

use crate::*;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul};

/// Internal Tensor2 multiplication trait
trait InternalMatmul: Sized {
    fn matmul(
        // An `m` by `k` row-major Tensor2.
        a: &[Self],
        // An `k` by `n` row-major Tensor2.
        b: &[Self],
        // An `m` by `n` row-major Tensor2.
        c: &mut [Self],
        // Rows of `a` and rows of `c`.
        m: usize,
        // Columns of `b` and columns of `c`.
        n: usize,
        // Columns of `a` and rows of `b`.
        k: usize,
    );
}
/// Default Tensor2 multiplication implementation.
impl<T: Debug + Mul<Output = T> + AddAssign + Copy + Debug> InternalMatmul for T {
    default fn matmul(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, k: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(c.len(), m * n);

        for l_index in 0..m {
            for m_index in 0..k {
                for n_index in 0..n {
                    let (i, j, k) = (
                        l_index * n + n_index,
                        l_index * k + m_index,
                        m_index * n + n_index,
                    );
                    c[i] += a[j] * b[k];
                }
            }
        }
    }
}
/// `f32` Tensor2 multiplication specialization.
impl InternalMatmul for f32 {
    fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), n * k);
        assert_eq!(c.len(), m * n);
        let (m, n, k) = (m as i32, n as i32, k as i32);
        unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m,
                n,
                k,
                1.,
                a,
                k,
                b,
                n,
                1.,
                c,
                n,
            );
        }
    }
}
/// `f64` Tensor2 multiplication specialization.
impl InternalMatmul for f64 {
    fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), n * k);
        assert_eq!(c.len(), m * n);
        let (m, n, k) = (m as i32, n as i32, k as i32);
        unsafe {
            cblas::dgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m,
                n,
                k,
                1.,
                a,
                k,
                b,
                n,
                1.,
                c,
                n,
            );
        }
    }
}

/// A trait for matrix multiplication.
pub trait Matmul<T> {
    type Output;
    /// ```text
    /// ┌───────┐        ┌─────┐  ┌─────┐
    /// │ 1 1 1 │        │ 1 2 │  │ 4 5 │
    /// │ 2 1 2 │.matmul(│ 2 1 │)=│ 6 9 │
    /// └───────┘        │ 1 2 │  └─────┘
    ///                  └─────┘
    /// ```
    /// - M: Rows of `self` and rows of `Self::Output`.
    /// - K: Columns of `self` and rows of `other`.
    /// - N: Columns of `other` and columns of `Self::Output`.
    fn matmul(&self, other: &T) -> Self::Output;
}

// Tensor2DxD
// --------------------------------------------------
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T>, const K: usize, const N: usize>
    Matmul<Tensor2SxS<T, K, N>> for Tensor2DxD<T>
where
    [(); K * N]:,
{
    type Output = Tensor2DxS<T, N>;
    fn matmul(&self, other: &Tensor2SxS<T, K, N>) -> Self::Output {
        assert_eq!(self.b, K, "Non-matching columns to rows");

        let m = self.a;
        let mut data = vec![Default::default(); N * m];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, N, K);
        Self::Output { data, a: m }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T>, const N: usize>
    Matmul<Tensor2DxS<T, N>> for Tensor2DxD<T>
{
    type Output = Tensor2DxS<T, N>;
    fn matmul(&self, other: &Tensor2DxS<T, N>) -> Self::Output {
        assert_eq!(self.b, other.a, "Non-matching columns to rows");

        let (m, k) = (self.a, self.b);
        let mut data = vec![Default::default(); m * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, N, k);
        Self::Output { data, a: m }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug, const K: usize>
    Matmul<Tensor2SxD<T, K>> for Tensor2DxD<T>
{
    type Output = Tensor2DxD<T>;
    fn matmul(&self, other: &Tensor2SxD<T, K>) -> Self::Output {
        let (m, n) = (self.a, other.b);
        let mut data = vec![Default::default(); m * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, n, K);
        Self::Output { data, a: m, b: n }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug>
    Matmul<Tensor2DxD<T>> for Tensor2DxD<T>
{
    type Output = Tensor2DxD<T>;
    fn matmul(&self, other: &Tensor2DxD<T>) -> Self::Output {
        assert_eq!(self.b, other.a, "Non-matching columns to rows");

        let (m, k, n) = (self.a, self.b, other.b);
        let mut data = vec![Default::default(); m * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, n, k);
        Self::Output { data, a: m, b: n }
    }
}
// Tensor2DxS
// --------------------------------------------------
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T>, const K: usize, const N: usize>
    Matmul<Tensor2SxS<T, K, N>> for Tensor2DxS<T, K>
where
    [(); K * N]:,
{
    type Output = Tensor2DxS<T, N>;
    fn matmul(&self, other: &Tensor2SxS<T, K, N>) -> Self::Output {
        let m = self.a;
        let mut data = vec![Default::default(); m * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, N, K);
        Self::Output { data, a: m }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T>, const K: usize, const N: usize>
    Matmul<Tensor2DxS<T, N>> for Tensor2DxS<T, K>
{
    type Output = Tensor2DxS<T, N>;
    fn matmul(&self, other: &Tensor2DxS<T, N>) -> Self::Output {
        let m = self.a;
        let mut data = vec![Default::default(); m * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, N, K);
        Self::Output { data, a: m }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug, const K: usize>
    Matmul<Tensor2SxD<T, K>> for Tensor2DxS<T, K>
{
    type Output = Tensor2DxD<T>;
    fn matmul(&self, other: &Tensor2SxD<T, K>) -> Self::Output {
        let (m, n) = (self.a, other.b);
        let mut data = vec![Default::default(); m * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, n, K);
        Self::Output { data, a: m, b: n }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug, const K: usize>
    Matmul<Tensor2DxD<T>> for Tensor2DxS<T, K>
{
    type Output = Tensor2DxD<T>;
    fn matmul(&self, other: &Tensor2DxD<T>) -> Self::Output {
        assert_eq!(K, other.a, "Non-matching columns to rows");

        let (m, n) = (self.a, other.b);
        let mut data = vec![Default::default(); m * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, m, n, K);
        Self::Output { data, a: m, b: n }
    }
}
// Tensor2SxD
// --------------------------------------------------
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T>,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<Tensor2SxS<T, K, N>> for Tensor2SxD<T, M>
where
    [(); M * N]:,
    [(); K * N]:,
{
    type Output = Tensor2SxS<T, M, N>;
    fn matmul(&self, other: &Tensor2SxS<T, K, N>) -> Self::Output {
        let mut data = vec![Default::default(); M * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, N, K);
        Self::Output { data }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T>, const M: usize, const N: usize>
    Matmul<Tensor2DxS<T, N>> for Tensor2SxD<T, M>
where
    [(); M * N]:,
{
    type Output = Tensor2SxS<T, M, N>;
    fn matmul(&self, other: &Tensor2DxS<T, N>) -> Self::Output {
        assert_eq!(self.b, other.a, "Non-matching columns to rows");

        let k = self.b;
        let mut data = vec![Default::default(); M * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, N, k);
        Self::Output { data }
    }
}
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug,
        const M: usize,
        const K: usize,
    > Matmul<Tensor2SxD<T, K>> for Tensor2SxD<T, M>
{
    type Output = Tensor2SxD<T, M>;
    fn matmul(&self, other: &Tensor2SxD<T, K>) -> Self::Output {
        let n = other.b;
        let mut data = vec![Default::default(); M * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, n, K);
        Self::Output { data, b: n }
    }
}
impl<T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug, const M: usize>
    Matmul<Tensor2DxD<T>> for Tensor2SxD<T, M>
{
    type Output = Tensor2SxD<T, M>;
    fn matmul(&self, other: &Tensor2DxD<T>) -> Self::Output {
        assert_eq!(self.b, other.a, "Non-matching columns to rows");

        let (k, n) = (self.b, other.b);
        let mut data = vec![Default::default(); M * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, n, k);
        Self::Output { data, b: n }
    }
}
// Tensor2SxS
// --------------------------------------------------
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T>,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<Tensor2SxS<T, K, N>> for Tensor2SxS<T, M, K>
where
    [(); M * K]:,
    [(); K * N]:,
    [(); M * N]:,
{
    type Output = Tensor2SxS<T, M, N>;
    fn matmul(&self, other: &Tensor2SxS<T, K, N>) -> Self::Output {
        let mut data = vec![Default::default(); M * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, N, K);
        Self::Output { data }
    }
}
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T>,
        const M: usize,
        const K: usize,
        const N: usize,
    > Matmul<Tensor2DxS<T, N>> for Tensor2SxS<T, M, K>
where
    [(); M * K]:,
    [(); M * N]:,
{
    type Output = Tensor2SxS<T, M, N>;
    fn matmul(&self, other: &Tensor2DxS<T, N>) -> Self::Output {
        let mut data = vec![Default::default(); M * N];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, N, K);
        Self::Output { data }
    }
}
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug,
        const M: usize,
        const K: usize,
    > Matmul<Tensor2SxD<T, K>> for Tensor2SxS<T, M, K>
where
    [(); M * K]:,
{
    type Output = Tensor2SxD<T, M>;
    fn matmul(&self, other: &Tensor2SxD<T, K>) -> Self::Output {
        let n = other.b;
        let mut data = vec![Default::default(); M * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, n, K);
        Self::Output { data, b: n }
    }
}
impl<
        T: Debug + Default + Copy + AddAssign + Mul<Output = T> + std::fmt::Debug,
        const M: usize,
        const K: usize,
    > Matmul<Tensor2DxD<T>> for Tensor2SxS<T, M, K>
where
    [(); M * K]:,
{
    type Output = Tensor2SxD<T, M>;
    fn matmul(&self, other: &Tensor2DxD<T>) -> Self::Output {
        assert_eq!(K, other.a, "Non-matching columns to rows");

        let n = other.b;
        let mut data = vec![Default::default(); M * n];
        InternalMatmul::matmul(&self.data, &other.data, &mut data, M, n, K);
        Self::Output { data, b: n }
    }
}
// Tests
// --------------------------------------------------
#[cfg(test)]
mod tests {
    use crate::*;
    use std::convert::TryFrom;
    // f32
    // --------------------------------------------------
    #[test]
    fn f32_dxd() {
        let a = Tensor2DxD::<f32>::try_from(vec![vec![1., 3., 5.], vec![2., 4., 6.]]).unwrap();
        let b =
            Tensor2DxD::<f32>::try_from(vec![vec![7., 10.], vec![8., 11.], vec![9., 12.]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2DxD::<f32>::try_from(vec![vec![76., 103.], vec![100., 136.]]).unwrap();
        assert_eq!(c, d);
    }
    // f64
    // --------------------------------------------------
    #[test]
    fn f64_dxd() {
        let a = Tensor2DxD::<f64>::try_from(vec![vec![1., 3., 5.], vec![2., 4., 6.]]).unwrap();
        let b =
            Tensor2DxD::<f64>::try_from(vec![vec![7., 10.], vec![8., 11.], vec![9., 12.]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2DxD::<f64>::try_from(vec![vec![76., 103.], vec![100., 136.]]).unwrap();
        assert_eq!(c, d);
    }
    // Tensor2DxD
    // --------------------------------------------------
    #[test]
    fn dxd_dxd() {
        let a = Tensor2DxD::try_from(vec![vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2DxD::try_from(vec![vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2DxD::try_from(vec![vec![76, 103], vec![100, 136]]).unwrap();
        assert_eq!(c, d);
    }
    #[test]
    fn dxd_dxs() {
        let a = Tensor2DxD::try_from(vec![vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2DxS::from(vec![[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2DxS::from(vec![[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn dxd_sxd() {
        let a = Tensor2DxD::try_from(vec![vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2SxD::try_from([vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxD::try_from([vec![76, 103], vec![100, 136]]).unwrap();
        assert_eq!(c, d);
    }
    #[test]
    fn dxd_sxs() {
        let a = Tensor2DxD::try_from(vec![vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2SxS::from([[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    // Tensor2DxS
    // --------------------------------------------------
    #[test]
    fn dxs_dxs() {
        let a = Tensor2DxS::from(vec![[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2DxS::from(vec![[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2DxS::from(vec![[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn dxs_sxd() {
        let a = Tensor2DxS::from(vec![[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2SxD::try_from([vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn dxs_dxd() {
        let a = Tensor2DxS::from(vec![[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2DxD::try_from(vec![vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2DxS::from(vec![[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn dxs_sxs() {
        let a = Tensor2DxS::from(vec![[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2SxS::from([[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    // Tensor2SxD
    // --------------------------------------------------
    #[test]
    fn sxd_sxd() {
        let a = Tensor2SxD::try_from([vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2SxD::try_from([vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxD::try_from([vec![76, 103], vec![100, 136]]).unwrap();
        assert_eq!(c, d);
    }
    #[test]
    fn sxd_dxs() {
        let a = Tensor2SxD::try_from([vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2DxS::from(vec![[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn sxd_sxs() {
        let a = Tensor2SxD::try_from([vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2SxS::from([[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn sxd_dxd() {
        let a = Tensor2SxD::try_from([vec![1, 3, 5], vec![2, 4, 6]]).unwrap();
        let b = Tensor2DxD::try_from(vec![vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxD::try_from([vec![76, 103], vec![100, 136]]).unwrap();
        assert_eq!(c, d);
    }
    // Tensor2SxS
    // --------------------------------------------------
    #[test]
    fn sxs_sxs() {
        let a = Tensor2SxS::<i32, 2, 3>::from([[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2SxS::<i32, 3, 2>::from([[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::<i32, 2, 2>::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn sxs_dxs() {
        let a = Tensor2SxS::from([[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2DxS::from(vec![[7, 10], [8, 11], [9, 12]]);
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn sxs_sxd() {
        let a = Tensor2SxS::from([[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2SxD::try_from([vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
    #[test]
    fn sxs_dxd() {
        let a = Tensor2SxS::from([[1, 3, 5], [2, 4, 6]]);
        let b = Tensor2DxD::try_from(vec![vec![7, 10], vec![8, 11], vec![9, 12]]).unwrap();
        let c = a.matmul(&b);
        let d = Tensor2SxS::from([[76, 103], [100, 136]]);
        assert_eq!(c, d);
    }
}
