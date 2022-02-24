#![allow(unstable_name_collisions)]
#![feature(format_args_capture)]

extern crate proc_macro;
use proc_macro::TokenStream;

use itertools::Itertools;

// a        A
// a*b      A*b, a*B, A*B
// a*b*c    A*b*c, a*B*c, a*b*C, A*B*c, a*B*C, A*b*C, A*B*C
// a*b*c*d  ...

// Tensor1D<T>, Tensor1S<T,A>
// Tensor2DD<T>, Tensor2SD<T,A>, Tensor2DS<T,B>, Tensor2SS<T,A,B>
// Tensor3DDD<T>, Tensor3SDD<T,A>, Tensor3DSD<T,B>, Tensor3DDS<T,C>, Tensor3SSD<T,A,B>, ..., Tensor3SSS<T,A,B,C>

// VectorD<T> = Tensor1D<T>
// VectorS<T> = Tensor1S<T>
// MatrixDD<T> = Tensor2DD<T>
// ...

/// Static dimension separator
const SDS: char = 'x';

/// Use this to avoid noise from `assert_eq!` expansion.
const ASSERT_EQ: &str = "assert_eq!";

#[proc_macro]
pub fn tensors(item: TokenStream) -> TokenStream {
    // The number of dimensions to define structs up to
    let dimensions = match item.into_iter().next().unwrap() {
        proc_macro::TokenTree::Literal(n) => n.to_string().parse::<usize>().unwrap(),
        _ => unreachable!(),
    };
    // eprintln!("dimensions: {}", dimensions);
    assert!(
        dimensions < 25,
        "Notation only allows dimensions up to 25 (A-Z)"
    );

    let mut out = String::new();
    // Tensor definitions
    // --------------------------------------------------------
    for i in 1..dimensions + 1 {
        // For all tensors of `i` dimensionality.
        for form in (0..i).map(|_| (0..2)).multi_cartesian_product() {
            // Definition
            // --------------------------------------
            let mut layout = (
                format!("pub struct Tensor{}", i.to_string()),
                vec![char::default(); i],
                vec![None; i],
                vec![None; i],
            );
            let form = form.into_iter().map(|x| x != 0).collect::<Vec<_>>();

            // For each dimension
            for c in (0..26).take(i) {
                if form[c] {
                    layout.1[c] = 'S';
                    layout.2[c] = Some((c as u8 + 65) as char);
                } else {
                    layout.1[c] = 'D';
                    layout.3[c] = Some(format!("pub {}: usize,", (c as u8 + 97) as char));
                }
            }

            // The dimensions description used in rustdoc
            let rustdoc_dims = layout
                .2
                .iter()
                .map(|x| match x {
                    Some(_) => "static",
                    None => "dynamic",
                })
                .intersperse(", ")
                .collect::<String>();

            // The static dimensions identifiers appended to the struct identifier.
            let static_dimensions = layout.1.iter().intersperse(&SDS).collect::<String>();

            // The static dimension const generics.
            let const_generics = layout
                .2
                .iter()
                .filter_map(|x| *x)
                .map(|x| format!(",const {}: usize", x))
                .collect::<String>();

            // The dynamic dimensions struct values.
            let dynamic_dimensions = layout.3.into_iter().filter_map(|x| x).collect::<String>();

            let type_info_t = format!("{}<T{}>", static_dimensions, const_generics);
            let type_info = format!(
                "{}<T{}>",
                static_dimensions,
                layout
                    .2
                    .iter()
                    .map(|x| match x {
                        Some(x) => format!(",{}", x),
                        None => String::from(""),
                    })
                    .collect::<String>()
            );
            // The potential alias for this tensor.
            let alias_string = if i == 1 {
                format!(
                    "
                    #[doc=\"An alias for a `[{}]` 1 dimensional tensor.\"]
                    pub type Vector{} = Tensor1{};",
                    rustdoc_dims, type_info_t, type_info
                )
            } else if i == 2 {
                format!(
                    "
                    #[doc=\"An alias for a `[{}]` 2 dimensional tensor.\"]
                    pub type Matrix{} = Tensor2{};
                    ",
                    rustdoc_dims, type_info_t, type_info
                )
            } else {
                String::from("")
            };

            // The full struct definition.
            #[allow(unstable_name_collisions)]
            let struct_string = format!(
                "
                #[doc=\"A {} dimensional tensor with `[{}]` dimensions.\"]
                {}{}<T{}>{{ pub data: Vec<T>, {} }}
                ",
                i, rustdoc_dims, layout.0, static_dimensions, const_generics, dynamic_dimensions,
            );

            // We push the struct definition.
            out.push_str(&struct_string);
            // eprintln!("{}", struct_string);

            // We push the alias string.
            out.push_str(&alias_string);

            // Implementations
            // impl_add(&mut out, i, &form, &type_info);
            // impl_sub(&mut out, i, &form, &type_info);
            // standard_impl(&mut out, i, &form, &type_info,"-","Sub","sub");
            standard_impl(&mut out, i, &form, &type_info,"+","Add","add");
        }
    }

    // Return
    // --------------------------------------------------------
    out.parse().unwrap()
}

/// Shared functionality for some similar operations (e.g. `std::ops::Sub`, `std::ops::Add`)
fn standard_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
    op: &str,
    op_trait: &str,
    op_fn: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = impl_form.into_iter().map(|x| x != 0).collect::<Vec<_>>();
        // Defines whether each dimension in our output is static or not
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { 'S' } else { 'D' })
            .intersperse(SDS)
            .collect::<String>();
        // Getting the const generics for our rhs tensor based off whether each dimension is static or not
        let rhs_const_generics = impl_form
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets all const generics definitions needed for `self` and `Rhs`
        let joined_const_generics = join
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(", const {}: usize", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets type definition of our output
        let new_type_definition = format!(
            "{}<T{}>",
            join.iter()
                .map(|x| if *x { 'S' } else { 'D' })
                .intersperse(SDS)
                .collect::<String>(),
            join.iter()
                .enumerate()
                .filter_map(|(v, x)| if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                })
                .collect::<String>()
        );
        let (mut self_mut, mut rhs_mut) = ("", "");
        // Type definition for `rhs`
        let rhs_definition = format!("{}<T{}>", impl_static_dimensions, rhs_const_generics);

        let mut assignment = format!(
            "
            let mut data = vec![Default::default();{}];
            for (a,(b,c)) in self.data.iter().zip(rhs.data.iter().zip(data.iter_mut())) {{
                *c = *a {op} *b;
            }}
            Self::Output {{ data {} }}
            ",
            // Dimensions lengths to construct underlying vec
            join.iter()
                .enumerate()
                .map(|(v, x)| if *x {
                    String::from((v as u8 + 65) as char)
                } else {
                    format!("self.{}", (v as u8 + 97) as char)
                })
                .intersperse(String::from("*"))
                .collect::<String>(),
            // Dynamic dimensions size parameters to set
            join.iter()
                .enumerate()
                .filter_map(|(v,x)|
                // If this dimension is not static in our output.
                if !*x {
                    // Then it is dynamic in both our inputs, thus both our inputs have the respective `(v as u8 + 97) as char` dimensions property.
                    // We pull the property from `self` rather than `rhs` out of simple preference.
                    let c = (v as u8 + 97) as char;
                    Some(format!(", {}: self.{}",c,c))
                }
                else {
                    None
                })
                .collect::<String>()
        );
        // Traits we need our in operation
        let mut needed_trait = format!("Default + Copy + std::ops::{op_trait}<Output=T>");
        // If our output is the same type as `self`
        if self_dimensionality_form == join {
            needed_trait = format!("Copy + std::ops::{op_trait}Assign");
            self_mut = "mut";
            assignment = format!(
                "
                for (a,b) in self.data.iter_mut().zip(rhs.data.iter()) {{ 
                    *a {op}= *b; 
                }}
                self
            ",
            );
        // If our output is the same type as `rhs`
        } else if impl_form == join {
            needed_trait = format!("Copy + std::ops::{op_trait}<Output=T>");
            rhs_mut = "mut";
            assignment = format!(
                "
                for (a,b) in self.data.iter().zip(rhs.data.iter_mut()) {{ 
                    *b = *a {op} *b; 
                }}
                rhs
            ",
            );
        }

        // Our full implementation block
        let impl_block = format!(
            "
            impl<T: {needed_trait}{joined_const_generics}> std::ops::{op_trait}<Tensor{ndims}{rhs_definition}> for Tensor{ndims}{} {{
                type Output = Tensor{ndims}{};
                fn {op_fn}({self_mut} self, {rhs_mut} rhs: Tensor{ndims}{rhs_definition}) -> Self::Output {{
                    {}
                    {assignment}
                }}
            }}
            ",
            // Type definition for `self`
            self_partial_type_suffix,
            new_type_definition,
            (0..26).take(ndims).filter_map(|d|
                // If both dimensions are static we don't need to check
                if self_dimensionality_form[d] && impl_form[d] {
                    None
                }
                // If one or both our input tensors are dynamic in a dimension we need to check their lengths
                else if self_dimensionality_form[d] {
                    Some(format!("{}({},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+65) as char,(d as u8+97) as char,d))
                }
                else if impl_form[d] {
                    Some(format!("{}(self.{},{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+65) as char,d))
                }
                else {
                    Some(format!("{}(self.{},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+97) as char,d))
                }
            ).collect::<String>(),
        );
        output_string.push_str(&impl_block);
    }
}

fn impl_add(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = impl_form.into_iter().map(|x| x != 0).collect::<Vec<_>>();
        // Defines whether each dimension in our output is static or not
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { 'S' } else { 'D' })
            .intersperse(SDS)
            .collect::<String>();
        // Getting the const generics for our rhs tensor based off whether each dimension is static or not
        let rhs_const_generics = impl_form
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets all const generics definitions needed for `self` and `Rhs`
        let joined_const_generics = join
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(", const {}: usize", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets type definition of our output
        let new_type_definition = format!(
            "{}<T{}>",
            join.iter()
                .map(|x| if *x { 'S' } else { 'D' })
                .intersperse(SDS)
                .collect::<String>(),
            join.iter()
                .enumerate()
                .filter_map(|(v, x)| if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                })
                .collect::<String>()
        );
        let (mut self_mut, mut rhs_mut) = ("", "");
        // Type definition for `rhs`
        let rhs_definition = format!("{}<T{}>", impl_static_dimensions, rhs_const_generics);

        let mut assignment = format!(
            "
            let mut data = vec![Default::default();{}];
            for (a,(b,c)) in self.data.iter().zip(rhs.data.iter().zip(data.iter_mut())) {{
                *c = *a + *b;
            }}
            Self::Output {{ data {} }}
            ",
            // Dimensions lengths to construct underlying vec
            join.iter()
                .enumerate()
                .map(|(v, x)| if *x {
                    String::from((v as u8 + 65) as char)
                } else {
                    format!("self.{}", (v as u8 + 97) as char)
                })
                .intersperse(String::from("*"))
                .collect::<String>(),
            // Dynamic dimensions size parameters to set
            join.iter()
                .enumerate()
                .filter_map(|(v,x)|
                // If this dimension is not static in our output.
                if !*x {
                    // Then it is dynamic in both our inputs, thus both our inputs have the respective `(v as u8 + 97) as char` dimensions property.
                    // We pull the property from `self` rather than `rhs` out of simple preference.
                    let c = (v as u8 + 97) as char;
                    Some(format!(", {}: self.{}",c,c))
                }
                else {
                    None
                })
                .collect::<String>()
        );
        // Traits we need our in operation
        let mut needed_trait = "Default + Copy + std::ops::Add<Output=T>";
        // If our output is the same type as `self`
        if self_dimensionality_form == join {
            needed_trait = "Copy + std::ops::AddAssign";
            self_mut = "mut";
            assignment = String::from(
                "
                for (a,b) in self.data.iter_mut().zip(rhs.data.iter()) { 
                    *a += *b; 
                }
                self
            ",
            );
        // If our output is the same type as `rhs`
        } else if impl_form == join {
            needed_trait = "Copy + std::ops::AddAssign";
            rhs_mut = "mut";
            assignment = String::from(
                "
                for (a,b) in self.data.iter().zip(rhs.data.iter_mut()) { 
                    *b += *a; 
                }
                rhs
            ",
            );
        }

        // Our full implementation block
        let impl_block = format!(
            "
            impl<T: {needed_trait}{joined_const_generics}> std::ops::Add<Tensor{ndims}{rhs_definition}> for Tensor{ndims}{} {{
                type Output = Tensor{ndims}{};
                fn add({self_mut} self, {rhs_mut} rhs: Tensor{ndims}{rhs_definition}) -> Self::Output {{
                    {}
                    {assignment}
                }}
            }}
            ",
            // Type definition for `self`
            self_partial_type_suffix,
            new_type_definition,
            (0..26).take(ndims).filter_map(|d|
                // If both dimensions are static we don't need to check
                if self_dimensionality_form[d] && impl_form[d] {
                    None
                }
                // If one or both our input tensors are dynamic in a dimension we need to check their lengths
                else if self_dimensionality_form[d] {
                    Some(format!("{}({},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+65) as char,(d as u8+97) as char,d))
                }
                else if impl_form[d] {
                    Some(format!("{}(self.{},{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+65) as char,d))
                }
                else {
                    Some(format!("{}(self.{},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+97) as char,d))
                }
            ).collect::<String>(),
        );
        output_string.push_str(&impl_block);
    }
}

fn impl_sub(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = impl_form.into_iter().map(|x| x != 0).collect::<Vec<_>>();
        // Defines whether each dimension in our output is static or not
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { 'S' } else { 'D' })
            .intersperse(SDS)
            .collect::<String>();
        // Getting the const generics for our rhs tensor based off whether each dimension is static or not
        let rhs_const_generics = impl_form
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets all const generics definitions needed for `self` and `Rhs`
        let joined_const_generics = join
            .iter()
            .enumerate()
            .filter_map(|(v, x)| {
                if *x {
                    Some(format!(", const {}: usize", (v as u8 + 65) as char))
                } else {
                    None
                }
            })
            .collect::<String>();
        // Gets type definition of our output
        let new_type_definition = format!(
            "{}<T{}>",
            join.iter()
                .map(|x| if *x { 'S' } else { 'D' })
                .intersperse(SDS)
                .collect::<String>(),
            join.iter()
                .enumerate()
                .filter_map(|(v, x)| if *x {
                    Some(format!(",{}", (v as u8 + 65) as char))
                } else {
                    None
                })
                .collect::<String>()
        );
        let (mut self_mut, mut rhs_mut) = ("", "");
        // Type definition for `rhs`
        let rhs_definition = format!("{}<T{}>", impl_static_dimensions, rhs_const_generics);

        let mut assignment = format!(
            "
            let mut data = vec![Default::default();{}];
            for (a,(b,c)) in self.data.iter().zip(rhs.data.iter().zip(data.iter_mut())) {{
                *c = *a - *b;
            }}
            Self::Output {{ data {} }}
            ",
            // Dimensions lengths to construct underlying vec
            join.iter()
                .enumerate()
                .map(|(v, x)| if *x {
                    String::from((v as u8 + 65) as char)
                } else {
                    format!("self.{}", (v as u8 + 97) as char)
                })
                .intersperse(String::from("*"))
                .collect::<String>(),
            // Dynamic dimensions size parameters to set
            join.iter()
                .enumerate()
                .filter_map(|(v,x)|
                // If this dimension is not static in our output.
                if !*x {
                    // Then it is dynamic in both our inputs, thus both our inputs have the respective `(v as u8 + 97) as char` dimensions property.
                    // We pull the property from `self` rather than `rhs` out of simple preference.
                    let c = (v as u8 + 97) as char;
                    Some(format!(", {}: self.{}",c,c))
                }
                else {
                    None
                })
                .collect::<String>()
        );
        // Traits we need our in operation
        let mut needed_trait = "Default + Copy + std::ops::Sub<Output=T>";
        // If our output is the same type as `self`
        if self_dimensionality_form == join {
            needed_trait = "Copy + std::ops::SubAssign";
            self_mut = "mut";
            assignment = String::from(
                "
                for (a,b) in self.data.iter_mut().zip(rhs.data.iter()) { 
                    *a -= *b; 
                }
                self
            ",
            );
        // If our output is the same type as `rhs`
        } else if impl_form == join {
            needed_trait = "Copy + std::ops::Sub<Output=T>";
            rhs_mut = "mut";
            assignment = String::from(
                "
                for (a,b) in self.data.iter().zip(rhs.data.iter_mut()) { 
                    *b = *a - *b; 
                }
                rhs
            ",
            );
        }

        // Our full implementation block
        let impl_block = format!(
            "
            impl<T: {needed_trait}{joined_const_generics}> std::ops::Sub<Tensor{ndims}{rhs_definition}> for Tensor{ndims}{} {{
                type Output = Tensor{ndims}{};
                fn sub({self_mut} self, {rhs_mut} rhs: Tensor{ndims}{rhs_definition}) -> Self::Output {{
                    {}
                    {assignment}
                }}
            }}
            ",
            // Type definition for `self`
            self_partial_type_suffix,
            new_type_definition,
            (0..26).take(ndims).filter_map(|d|
                // If both dimensions are static we don't need to check
                if self_dimensionality_form[d] && impl_form[d] {
                    None
                }
                // If one or both our input tensors are dynamic in a dimension we need to check their lengths
                else if self_dimensionality_form[d] {
                    Some(format!("{}({},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+65) as char,(d as u8+97) as char,d))
                }
                else if impl_form[d] {
                    Some(format!("{}(self.{},{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+65) as char,d))
                }
                else {
                    Some(format!("{}(self.{},rhs.{},\"Dimension {} of the given tensors doesn't match\");",ASSERT_EQ,(d as u8+97) as char,(d as u8+97) as char,d))
                }
            ).collect::<String>(),
        );
        output_string.push_str(&impl_block);
    }
}
