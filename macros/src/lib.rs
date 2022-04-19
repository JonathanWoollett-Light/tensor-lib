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

/// Static dimension separator.
const SDS: &str = "x";
/// Static dimension static tag.
const SDST: &str = "S";
/// Static dimension dynamic tag.
const SDDT: &str = "D";

const RANGE_SUFFIX: &str = "_range";

const DEFAULT_LEVEL: usize = 2;

/// Use this to avoid noise from `assert_eq!` expansion.
const ASSERT_EQ: &str = "assert_eq!";
const LOWERCASE_OFFSET: u8 = 97;
const UPPERCASE_OFFSET: u8 = 65;
// Converts integer in range 0..27 to a lowercase letter (e.g. 1->b).
fn lowercase(x: usize) -> char {
    (x as u8 + LOWERCASE_OFFSET) as char
}
// Converts integer in range 0..27 to an uppercase letter (e.g. 1->B).
fn uppercase(x: usize) -> char {
    (x as u8 + UPPERCASE_OFFSET) as char
}

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
    blas(&mut out);
    // For each dimension
    for i in 1..dimensions + 1 {
        let join_trait = format!(
            "
            #[doc=\"A trait defining tensors which can be joined along dimension {i}.\"]
            pub trait Join{i}<T> {{
                type Output;
                #[doc=\"Joins `self` and `rhs` along dimension {i} forming `Self::Output`.\"]
                fn join{i}(self, rhs: T)-> Self::Output;
            }}
            "
        );
        out.push_str(&join_trait);

        // For all permutations of dynamic & static tensors of `i` dimensionality.
        for form in (0..i).map(|_| (0..2)).multi_cartesian_product() {
            // Convert form to boolean (1->true, 0->false).
            let form = bool_vec(form);
            // Definition
            // --------------------------------------
            let mut layout = (
                format!("pub struct Tensor{}", i),
                vec![char::default(); i],
                vec![None; i],
                vec![None; i],
            );

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
            let rustdoc_dims = form
                .iter()
                .map(|b| if *b { "static" } else { "dynamic" })
                .intersperse(", ")
                .collect::<String>();

            // The static dimensions identifiers appended to the struct identifier.
            let static_dimensions = form
                .iter()
                .map(|b| if *b { SDST } else { SDDT })
                .intersperse(SDS)
                .collect::<String>();

            // The static dimension const generics.
            let const_generics = form
                .iter()
                .enumerate()
                // Filter for only the static dimensions
                .filter(|(_, b)| **b)
                .map(|(v, _)| format!(", const {}: usize", uppercase(v)))
                .collect::<String>();

            // The dynamic dimensions struct values.
            let dynamic_dimensions = form
                .iter()
                .enumerate()
                // Filter for only the dynamic dimensions
                .filter(|(_, b)| !**b)
                .map(|(v, _)| format!("pub {}: usize,", lowercase(v)))
                .collect::<String>();

            let type_info_t = format!("{}<T{}>", static_dimensions, const_generics);
            let static_type_dimensions = form
                .iter()
                .enumerate()
                // Filter for only the static dimensions
                .filter(|(_, b)| **b)
                .map(|(v, _)| format!(",{}", uppercase(v)))
                .collect::<String>();
            let type_info = format!("{}<T{}>", static_dimensions, static_type_dimensions);

            let mut features = if i <= DEFAULT_LEVEL {
                String::from("feature=\"default\",")
            } else {
                String::new()
            };
            features.push_str(
                &(i..=dimensions)
                    .map(|v| format!("feature=\"d{v}\""))
                    .intersperse(String::from(","))
                    .collect::<String>(),
            );

            // The low dimensionality linear algebra specific functionality
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
                    #[doc=\"An alias for a `[{rustdoc_dims}]` 2 dimensional tensor.\"]
                    pub type Matrix{type_info_t} = Tensor2{type_info};
                    #[doc=\"Transposed 2d tensor with `[{rustdoc_dims}]` dimensions.\"]
                    #[derive(Debug,Clone)]
                    pub struct Transpose{type_info_t}(pub Tensor2{type_info});
                    "
                )
            } else {
                String::new()
            };

            // The full struct definition.
            // #[derive(Eq,PartialEq,Debug,Clone)]
            #[allow(unstable_name_collisions)]
            let struct_string = format!(
                "
                #[doc=\"{i}d tensor with `[{rustdoc_dims}]` dimensions.\n\n`self.data` contains the underlying data.\n\nIf present the properties `a`, `b`, `c`, etc. represent the lengths of dimensions `0`, `1`, `2`, etc.\"]
                #[derive(Eq,PartialEq,Debug,Clone)]
                pub struct Tensor{i}{static_dimensions}<T{const_generics}>{{ pub data: Vec<T>, {dynamic_dimensions} }}
                impl<T{const_generics}> Tensor{i}{type_info}{{
                    pub fn iter(&self) -> impl Iterator<Item=&T> {{
                        self.data.iter()
                    }}
                    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut T> {{
                        self.data.iter_mut()
                    }}
                    pub fn len(&self) -> usize {{
                        self.data.len()
                    }}
                    pub fn is_empty(&self) -> bool {{
                        self.data.is_empty()
                    }}
                    {}
                }}
                ", if i==2 {
                    // eprintln!("type_info: {}",type_info);
                    // let trans = format!("");
                    // eprintln!("trans: {}",trans);
                    format!(
                        "
                        pub fn transpose(self) -> Transpose{type_info} {{
                            Transpose{static_dimensions}(self)
                        }}
                        "
                    )
                } else {
                    String::new()
                }
            );

            // We push the struct definition.
            out.push_str(&struct_string);
            // eprintln!("{}", struct_string);

            // We push the alias string.
            out.push_str(&alias_string);

            // We push specific type traits.
            let f32_type = format!("{}<f32{}>", static_dimensions, static_type_dimensions);
            let f64_type = format!("{}<f64{}>", static_dimensions, static_type_dimensions);
            // Strips starting command from `const_generics` since we don't expect a `T` generic value to proceed its usage here.
            let non_t_const_generics = if const_generics.is_empty() {
                &const_generics
            } else {
                &const_generics[1..]
            };

            out.push_str(&format!(
                "
                impl<{non_t_const_generics}> Tensor<f32> for Tensor{i}{f32_type} {{
                    fn data(&self) -> &[f32] {{
                        &self.data
                    }}
                    fn data_mut(&mut self) -> &mut [f32] {{
                        &mut self.data
                    }}
                }}
                impl<{non_t_const_generics}> Tensor<f64> for Tensor{i}{f64_type} {{
                    fn data(&self) -> &[f64] {{
                        &self.data
                    }}
                    fn data_mut(&mut self) -> &mut [f64] {{
                        &mut self.data
                    }}
                }}
                "
            ));

            // Slicing trait
            // --------------------------------------
            // Create slicing trait for this permutation output
            let const_slice_ranges = form
                .iter()
                .enumerate()
                // Filter for only the static dimensions
                .filter(|(_, b)| **b)
                .map(|(v, _)| {
                    format!(
                        "const {}{RANGE_SUFFIX}: std::ops::Range<usize>",
                        uppercase(v)
                    )
                })
                .intersperse(String::from(","))
                .collect::<String>();
            let dynamic_slice_ranges = form
                .iter()
                .enumerate()
                // Filter for only the dynamic dimensions
                .filter(|(_, b)| !**b)
                .map(|(v, _)| format!(",{}{RANGE_SUFFIX}: std::ops::Range<usize>", lowercase(v)))
                .collect::<String>();
            let const_slice_sizes = form
                .iter()
                .enumerate()
                // Filter for only the static dimensions
                .filter(|(_, b)| **b)
                .map(|(v, _)| format!(",{{range_len({}{RANGE_SUFFIX})}}", uppercase(v)))
                .collect::<String>();
            let slice_trait_str = format!(
                "
                #[doc=\"Support for slicing a {i}d tensor by `[{rustdoc_dims}]` ranges.\"]
                pub trait Slice{i}{static_dimensions}<T> {{
                    fn slice<{const_slice_ranges}>(&self{dynamic_slice_ranges}) -> Tensor{i}{static_dimensions}<&T{const_slice_sizes}>;
                }}
                "
            );

            // We push our slice trait
            out.push_str(&slice_trait_str);

            // Atomic implementations
            // --------------------------------------
            // Implementations which only depend on the type of `self`

            // BLAS
            blas_impl(&mut out, i, &form);
            // Join
            join_impl(&mut out, i, &form, &type_info);
            // Display
            display_impl(&mut out, i, &form, &type_info);
            // From
            from(&mut out, i, &form, &type_info);
            from_distribution(&mut out, i, &form, &type_info);
            // Slicing
            slice_impl(&mut out, i, &form, &static_dimensions, &const_generics);
            // Index
            index(&mut out, i, &form, &type_info);
            // Basic ops
            standard_impl(&mut out, i, &form, &type_info, "+", "Add", "add");
            standard_impl(&mut out, i, &form, &type_info, "-", "Sub", "sub");
            standard_impl(&mut out, i, &form, &type_info, "*", "Mul", "mul");
            standard_impl(&mut out, i, &form, &type_info, "/", "Div", "div");
            standard_impl(&mut out, i, &form, &type_info, "|", "BitOr", "bitor");
            standard_impl(&mut out, i, &form, &type_info, "&", "BitAnd", "bitand");
            standard_impl(&mut out, i, &form, &type_info, "^", "BitXor", "bitxor");
            standard_impl(&mut out, i, &form, &type_info, "%", "Rem", "rem");
            // Assign ops
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "+=",
                "AddAssign",
                "add_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "-=",
                "SubAssign",
                "sub_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "*=",
                "MulAssign",
                "mul_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "/=",
                "DivAssign",
                "div_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "|=",
                "BitOrAssign",
                "bitor_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "&=",
                "BitAndAssign",
                "bitand_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "^=",
                "BitXorAssign",
                "bitxor_assign",
            );
            assign_impl(
                &mut out,
                i,
                &form,
                &type_info,
                "%=",
                "RemAssign",
                "rem_assign",
            );
        }
    }

    // Return
    // --------------------------------------------------------
    out.parse().unwrap()
}
/// Converts a vector of `0` and `1` into a vector of `false` and `true` (respectively).
fn bool_vec(x: Vec<u8>) -> Vec<bool> {
    x.into_iter().map(|x| x != 0).collect()
}
fn display_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    // [A, b, C, D, ...]
    let idents = self_dimensionality_form
        .iter()
        .enumerate()
        .map(|(v, x)| {
            if *x {
                (uppercase(v), *x)
            } else {
                (lowercase(v), *x)
            }
        })
        .collect::<Vec<_>>();

    let (static_idents, _) = idents.iter().cloned().partition::<Vec<_>, _>(|(_, x)| *x);

    let idents = idents
        .into_iter()
        .map(|(v, x)| {
            if x {
                v.to_string()
            } else {
                format!("self.{}", v)
            }
        })
        .collect::<Vec<_>>();
    let horizontal_idents = idents.iter().cloned().step_by(2).collect::<Vec<_>>();
    let vertical_idents = idents
        .iter()
        .cloned()
        .skip(1)
        .step_by(2)
        .collect::<Vec<_>>();

    // [...,E,self.c,A]
    let horizontal_idents_slice = horizontal_idents
        .iter()
        .rev()
        .map(|v| format!("{},", v))
        .collect::<String>();

    // [const A:usize, const B:usize, ...]
    let static_idents = static_idents
        .into_iter()
        .map(|(v, _)| format!(",const {}:usize", v))
        .collect::<String>();

    let idents_index = format!(
        "[{}]",
        (0..ndims)
            .map(|v| format!("{},", lowercase(v)))
            .collect::<String>()
    );

    let impl_block = format!(
        "
        impl<T: std::fmt::Display + std::fmt::Debug{static_idents}> std::fmt::Display for Tensor{ndims}{self_partial_type_suffix} {{
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{
                let max_str_width = self.data
                    .iter()
                    .map(|v|v.to_string().chars().count())
                    .max()
                    .unwrap_or(0);
                let (upper_bounds,lower_bounds) = display_bounds(&[{horizontal_idents_slice}],max_str_width);

                let mut out = String::new();
                {}
                write!(f,\"\\n{{}}\",out)
            }}
        }}
        ",
        vertical_dimensions(
            &vertical_idents,
            &horizontal_idents,
            idents_index,
        )
    );
    output_string.push_str(&impl_block);

    fn vertical_dimensions(
        // [self.b,D,self.f,...].rev()
        v_dims: &[String],
        // [A,C,self.e,...].rev()
        h_dims: &[String],
        // "[a,b,c,d,e,f,...]"
        index: String,
    ) -> String {
        // Assume in-order not reversed, h_dims=[A,self.c,E,...]
        let mut outwards = format!(
            "
            out.push('│');
            for a in 0..{} {{
                out.push(' ');
                let s = self[{index}].to_string();
                let v = format!(\"{{}}{{}}\",\" \".repeat(max_str_width-s.chars().count()),s);
                out.push_str(&v);
            }}
            out.push(' ');
            out.push('│');
            ",
            h_dims[0]
        );
        for (dim, index) in h_dims.iter().zip((0..).step_by(2)).skip(1) {
            let ident = lowercase(index);
            outwards = format!(
                "
                out.push('│');
                for {ident} in 0..{dim} {{
                    {outwards}
                }}
                out.push('│');
                "
            );
        }

        let mut iter = v_dims.iter().zip((1..).step_by(2));

        if let Some((first_dim, _)) = iter.next() {
            outwards = format!(
                "
                for b in 0..{first_dim} {{
                    {outwards}
                    if b+1 < {first_dim} {{
                        out.push('\\n');
                    }}
                }}
                "
            )
        }
        let mut bound_iter = (1..h_dims.len()).rev();
        loop {
            if let Some(next_bound) = bound_iter.next() {
                outwards = format!(
                    "
                    out.push_str(&upper_bounds[{next_bound}]);
                    out.push('\\n');
                    {outwards}
                    out.push('\\n');
                    out.push_str(&lower_bounds[{next_bound}]);
                    "
                );
            } else {
                break;
            }
            if let Some((dim, index)) = iter.next() {
                let ident = lowercase(index);
                outwards = format!(
                    "
                    
                    for {ident} in 0..{dim} {{
                        {outwards}
                        if {ident}+1 < {dim} {{
                            out.push('\\n');
                        }}
                    }}
                    "
                );
            }
        }
        outwards = format!(
            "
            out.push_str(&upper_bounds[0]);
            out.push('\\n');
            {outwards}
            out.push('\\n');
            out.push_str(&lower_bounds[0]);
            "
        );
        // eprintln!("index: {}, outwards: {}",index,outwards);

        outwards
    }
}
fn join_impl(
    output_string: &mut String,
    ndims: usize,
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = bool_vec(impl_form);

        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { SDST } else { SDDT })
            .intersperse(SDS)
            .collect::<String>();
        for i in 0..ndims {
            // Gets all const generics definitions needed for `self` and `Rhs`
            let joined_const_generics = self_dimensionality_form
                .iter()
                .zip(impl_form.iter())
                .enumerate()
                .map(|(v, (a, b))| {
                    let mut s = String::new();
                    let t = (v as u8 + 65) as char;
                    if *a {
                        s.push_str(&format!(", const {t}: usize"));
                    }
                    // If this dimensions is static in `rhs`
                    if *b {
                        // If this is the dimension we are joining along
                        if v == i {
                            s.push_str(&format!(", const X{t}: usize"));
                        }
                        // If this dimension was not static in `self` but is static in `rhs`
                        else if !*a {
                            s.push_str(&format!(", const {t}: usize"))
                        }
                    }
                    s
                })
                .collect::<String>();
            // Getting the const generics for our rhs tensor based off whether each dimension is static or not
            let rhs_const_generics = impl_form
                .iter()
                .enumerate()
                .filter_map(|(v, x)| {
                    if *x {
                        if v == i {
                            Some(format!(",X{}", (v as u8 + 65) as char))
                        } else {
                            Some(format!(",{}", (v as u8 + 65) as char))
                        }
                    } else {
                        None
                    }
                })
                .collect::<String>();
            // Type definition for `rhs`
            let rhs_definition =
                format!("Tensor{ndims}{impl_static_dimensions}<T{rhs_const_generics}>");

            // When joining along the i'th dimension if either dimensions in `self` or `other` is
            //  static they remain static unless it is the i'th dimension in which case it is only
            //  static if it is static in both `self` and `other`.
            let out_form = self_dimensionality_form
                .iter()
                .zip(impl_form.iter())
                .enumerate()
                .map(
                    |(index, (a, b))| {
                        if index == i {
                            *a && *b
                        } else {
                            *a || *b
                        }
                    },
                )
                .collect::<Vec<_>>();
            // Getting the suffix for our Self::Output tensor based off whether each dimensions is static or not
            let out_static_dimensions = out_form
                .iter()
                .map(|x| if *x { SDST } else { SDDT })
                .intersperse(SDS)
                .collect::<String>();
            // Getting the const generics for our Self::Output tensor based off whether each dimension is static or not
            let out_const_generics = out_form
                .iter()
                .enumerate()
                .filter_map(|(v, x)| {
                    if *x {
                        let t = (v as u8 + 65) as char;
                        if v == i {
                            Some(format!(",{{ {t}+X{t} }}"))
                        } else {
                            Some(format!(",{t}"))
                        }
                    } else {
                        None
                    }
                })
                .collect::<String>();
            // Type definition for `Self::Output`
            let out_definition =
                format!("Tensor{ndims}{out_static_dimensions}<T{out_const_generics}>");

            let out_dynamic_dimensions = out_form
                .iter()
                .enumerate()
                // Filter out static dimensions
                .filter_map(|(v, b)| if *b { None } else { Some(v) })
                .map(|v| {
                    format!(",{}: {}", lowercase(v), {
                        let self_size = if self_dimensionality_form[v] {
                            String::from(uppercase(v))
                        } else {
                            format!("self.{}", lowercase(v))
                        };
                        if v == i {
                            let rhs_size = if impl_form[v] {
                                format!("X{}", uppercase(v))
                            } else {
                                format!("rhs.{}", lowercase(v))
                            };
                            format!("{self_size}+{rhs_size}")
                        } else {
                            self_size
                        }
                    })
                })
                .collect::<String>();

            let self_chunk_size = self_dimensionality_form
                .iter()
                .enumerate()
                .take(i + 1)
                .map(|(v, b)| {
                    if *b {
                        String::from(uppercase(v))
                    } else {
                        format!("self.{}", lowercase(v))
                    }
                })
                .intersperse(String::from("*"))
                .collect::<String>();
            let rhs_chunk_size = impl_form
                .iter()
                .enumerate()
                .take(i + 1)
                .map(|(v, b)| {
                    if *b {
                        if v == i {
                            format!("X{}", uppercase(v))
                        } else {
                            String::from(uppercase(v))
                        }
                    } else {
                        format!("rhs.{}", lowercase(v))
                    }
                })
                .intersperse(String::from("*"))
                .collect::<String>();

            let checks = self_dimensionality_form
                .iter()
                .zip(impl_form.iter())
                .enumerate()
                .map(|(v, (a, b))| {
                    // The dimension we are joining along does not need to match
                    if v == i {
                        String::new()
                    } else {
                        match (a, b) {
                            (true, true) => String::new(),
                            (false, true) => {
                                format!(
                                    "assert_eq!(self.{},{},\"Bad tensor sizes\");",
                                    lowercase(v),
                                    uppercase(v)
                                )
                            }
                            (true, false) => {
                                format!(
                                    "assert_eq!({},rhs.{},\"Bad tensor sizes\");",
                                    uppercase(v),
                                    lowercase(v)
                                )
                            }
                            (false, false) => {
                                format!(
                                    "assert_eq!(self.{},rhs.{},\"Bad tensor sizes\");",
                                    lowercase(v),
                                    lowercase(v)
                                )
                            }
                        }
                    }
                })
                .collect::<String>();
            let where_constraints = if self_dimensionality_form[i] && impl_form[i] {
                format!("where [();{} + X{}]:", uppercase(i), uppercase(i))
            } else {
                String::new()
            };
            // eprintln!("where_constraints: {}",where_constraints);

            let impl_block = format!(
                "
                impl<T:Clone{joined_const_generics}> Join{}<{rhs_definition}> for Tensor{ndims}{self_partial_type_suffix} {where_constraints} {{
                    type Output = {out_definition};
                    fn join{}(self, rhs: {rhs_definition}) -> Self::Output {{
                        {checks}
                        Self::Output {{
                            data: self
                                .data
                                .chunks_exact({self_chunk_size})
                                .zip(rhs.data.chunks_exact({rhs_chunk_size}))
                                .flat_map(|(a,b)| [a,b].concat())
                                .collect()
                            {out_dynamic_dimensions}
                        }}
                    }}
                }}
                ",
                i+1,
                i+1
            );
            // eprintln!("impl_block: {}",impl_block);
            // eprintln!("impl_block: {}",impl_block);
            output_string.push_str(&impl_block);
        }
    }
}
/// Generates simple blas implementations
fn blas_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
) {
    let const_dims = self_dimensionality_form
        .iter()
        .enumerate()
        .filter_map(|(v, x)| x.then(|| uppercase(v)))
        .collect::<Vec<_>>();
    let const_generics = const_dims
        .iter()
        .map(|v| format!(", {}", v))
        .collect::<String>();

    let static_dims = self_dimensionality_form
        .iter()
        .map(|b| if *b { SDST } else { SDDT })
        .intersperse(SDS)
        .collect::<String>();

    let single_type = format!("Tensor{ndims}{static_dims}<f32{const_generics}>");
    let double_type = format!("Tensor{ndims}{static_dims}<f64{const_generics}>");
    // eprintln!("single_type: {}",single_type);

    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = bool_vec(impl_form);
        // Defines all the static dimensions
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        let static_idents = join
            .iter()
            .enumerate()
            .filter_map(|(v, b)| b.then(|| format!("const {}: usize, ", uppercase(v))))
            .collect::<String>();
        let impl_const_dims = impl_form
            .iter()
            .enumerate()
            .filter_map(|(v, b)| b.then(|| uppercase(v)))
            .collect::<Vec<_>>();
        let impl_const_generics = impl_const_dims
            .iter()
            .map(|v| format!(", {}", v))
            .collect::<String>();
        let impl_static_dims = impl_form
            .iter()
            .map(|b| if *b { SDST } else { SDDT })
            .intersperse(SDS)
            .collect::<String>();
        let impl_single_type = format!("Tensor{ndims}{impl_static_dims}<f32{impl_const_generics}>");
        let impl_double_type = format!("Tensor{ndims}{impl_static_dims}<f64{impl_const_generics}>");

        // eprintln!("static_idents: {}, impl_single_type: {}",static_idents,impl_single_type);

        let impl_block = format!(
            "
            impl<{static_idents}> SSWAP<{single_type},{impl_single_type}> for BLAS {{
                fn sswap(x: &mut {single_type}, y: &mut {impl_single_type}) {{
                    sswap(&mut x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> DSWAP<{double_type},{impl_double_type}> for BLAS {{
                fn dswap(x: &mut {double_type}, y: &mut {impl_double_type}) {{
                    dswap(&mut x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> SCOPY<{single_type},{impl_single_type}> for BLAS {{
                fn scopy(x: &{single_type}, y: &mut {impl_single_type}) {{
                    scopy(&x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> DCOPY<{double_type},{impl_double_type}> for BLAS {{
                fn dcopy(x: &{double_type}, y: &mut {impl_double_type}) {{
                    dcopy(&x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> SAXPY<{single_type},{impl_single_type}> for BLAS {{
                fn saxpy(alpha: f32, x: &{single_type}, y: &mut {impl_single_type}) {{
                    saxpy(alpha, &x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> DAXPY<{double_type},{impl_double_type}> for BLAS {{
                fn daxpy(alpha:f64, x: &{double_type}, y: &mut {impl_double_type}) {{
                    daxpy(alpha, &x.data, &mut y.data);
                }}
            }}
            impl<{static_idents}> SDOT<{single_type},{impl_single_type}> for BLAS {{
                fn sdot(x: &{single_type}, y: &{impl_single_type}) -> f32 {{
                    sdot(&x.data, &y.data)
                }}
            }}
            impl<{static_idents}> DDOT<{double_type},{impl_double_type}> for BLAS {{
                fn ddot(x: &{double_type}, y: &{impl_double_type}) -> f64 {{
                    ddot(&x.data, &y.data)
                }}
            }}
            "
        );
        output_string.push_str(&impl_block);
    }
}
/// Generate complex blas implementations
fn blas(out: &mut String) {
    // let a_dims = ['M','K'];
    // let b_dims = ['K','N'];
    // let c_dims = ['M','N'];

    // [M,N,K]

    // sgemv & dgemv

    // sgemm & dgemm
    // -------------------------------------------
    for a_form in vec![(0..2), (0..2)].into_iter().multi_cartesian_product() {
        let a_form = bool_vec(a_form);
        let a_present = [a_form[0], false, a_form[1]];

        let a_static_dims = a_form
            .iter()
            .map(|b| if *b { SDST } else { SDDT })
            .intersperse(SDS)
            .collect::<String>();
        let a_const_generics = match (a_form[0], a_form[1]) {
            (false, false) => "",
            (true, false) => ",M",
            (false, true) => ",K",
            (true, true) => ",M,K",
        };
        let dgemm_a_type = format!("Matrix{a_static_dims}<f64{a_const_generics}>");
        let sgemm_a_type = format!("Matrix{a_static_dims}<f32{a_const_generics}>");

        for b_form in vec![(0..2), (0..2)].into_iter().multi_cartesian_product() {
            let b_form = bool_vec(b_form);

            // [M,N,K]
            let b_present = [
                a_present[0], // || false
                a_present[1] || b_form[1],
                a_present[2] || b_form[0],
            ];

            let b_static_dims = b_form
                .iter()
                .map(|b| if *b { SDST } else { SDDT })
                .intersperse(SDS)
                .collect::<String>();
            let b_const_generics = match (b_form[0], b_form[1]) {
                (false, false) => "",
                (true, false) => ",K",
                (false, true) => ",N",
                (true, true) => ",K,N",
            };
            let dgemm_b_type = format!("Matrix{b_static_dims}<f64{b_const_generics}>");
            let sgemm_b_type = format!("Matrix{b_static_dims}<f32{b_const_generics}>");

            for c_form in vec![(0..2), (0..2)].into_iter().multi_cartesian_product() {
                let c_form = bool_vec(c_form);

                // [M,N,K]
                let c_present = [
                    b_present[0] || c_form[0],
                    b_present[1] || c_form[1],
                    b_present[2], // || false
                ];

                let c_static_dims = c_form
                    .iter()
                    .map(|b| if *b { SDST } else { SDDT })
                    .intersperse(SDS)
                    .collect::<String>();
                let c_const_generics = match (c_form[0], c_form[1]) {
                    (false, false) => "",
                    (true, false) => ",M",
                    (false, true) => ",N",
                    (true, true) => ",M,N",
                };
                let sgemm_c_type = format!("Matrix{c_static_dims}<f32{c_const_generics}>");
                let dgemm_c_type = format!("Matrix{c_static_dims}<f64{c_const_generics}>");

                let mut defined_const_generics = String::new();
                if c_present[0] {
                    defined_const_generics.push_str("const M: usize,");
                }
                if c_present[1] {
                    defined_const_generics.push_str("const N: usize,");
                }
                if c_present[2] {
                    defined_const_generics.push_str("const K: usize,");
                }

                let sgemm_impl_str = format!(
                    "
                    impl<{defined_const_generics}> SGEMM<{sgemm_a_type},{sgemm_b_type},{sgemm_c_type}> 
                    for BLAS {{
                        fn sgemm(
                            a: &{sgemm_a_type}, 
                            b: &{sgemm_b_type}, 
                            c: &mut {sgemm_c_type}, 
                            alpha: f32, 
                            beta: f32
                        ) {{
                            sgemm(false,false,{},{},{},alpha,beta,&a.data,&b.data,&mut c.data);
                        }}
                    }}
                    ",
                    if c_present[0] { "M" } else { "a.a" },
                    if c_present[1] { "N" } else { "b.b" },
                    if c_present[2] { "K" } else { "a.b" }
                );
                out.push_str(&sgemm_impl_str);
                let dgemm_impl_str = format!(
                    "
                    impl<{defined_const_generics}> DGEMM<{dgemm_a_type},{dgemm_b_type},{dgemm_c_type}> 
                    for BLAS {{
                        fn dgemm(
                            a: &{dgemm_a_type}, 
                            b: &{dgemm_b_type}, 
                            c: &mut {dgemm_c_type}, 
                            alpha: f64, 
                            beta: f64
                        ) {{
                            dgemm(false,false,{},{},{},alpha,beta,&a.data,&b.data,&mut c.data);
                        }}
                    }}
                    ",
                    if c_present[0] { "M" } else { "a.a" },
                    if c_present[1] { "N" } else { "b.b" },
                    if c_present[2] { "K" } else { "a.b" }
                );
                out.push_str(&dgemm_impl_str);
            }
        }
    }
}
fn from(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    let (static_idents, dynamic_idents) = self_dimensionality_form
        .iter()
        .enumerate()
        .map(|(v, x)| {
            if *x {
                ((v as u8 + 65) as char, *x)
            } else {
                ((v as u8 + 97) as char, *x)
            }
        })
        .partition::<Vec<_>, _>(|(_, x)| *x);
    let (static_idents, dynamic_idents) = (
        static_idents
            .into_iter()
            .map(|(v, _)| v)
            .collect::<Vec<_>>(),
        dynamic_idents
            .into_iter()
            .map(|(v, _)| v)
            .collect::<Vec<_>>(),
    );

    // Our full implementation block
    let impl_block = format!(
        "
        impl<T{}> From<({}Vec<T>)> for Tensor{ndims}{self_partial_type_suffix} {{
            fn from(({}data):({}Vec<T>)) -> Self {{
                assert_eq!({},data.len());
                Self {{ {}data }}
            }}
        }}
        ",
        static_idents
            .iter()
            .map(|v| format!(",const {}:usize", v))
            .collect::<String>(),
        "usize,".repeat(dynamic_idents.len()),
        dynamic_idents
            .iter()
            .map(|v| format!("{},", v))
            .collect::<String>(),
        "usize,".repeat(dynamic_idents.len()),
        static_idents
            .iter()
            .chain(dynamic_idents.iter())
            .intersperse(&'*')
            .collect::<String>(),
        dynamic_idents
            .iter()
            .map(|v| format!("{},", v))
            .collect::<String>(),
    );
    output_string.push_str(&impl_block);
}
fn from_distribution(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    let (static_idents, dynamic_idents) = self_dimensionality_form
        .iter()
        .enumerate()
        .map(|(v, x)| {
            if *x {
                ((v as u8 + 65) as char, *x)
            } else {
                ((v as u8 + 97) as char, *x)
            }
        })
        .partition::<Vec<_>, _>(|(_, x)| *x);
    let (static_idents, dynamic_idents) = (
        static_idents
            .into_iter()
            .map(|(v, _)| v)
            .collect::<Vec<_>>(),
        dynamic_idents
            .into_iter()
            .map(|(v, _)| v)
            .collect::<Vec<_>>(),
    );

    let dynamic_shape = (2..)
        .take(dynamic_idents.len())
        .map(|v| format!("{},", v))
        .collect::<String>();
    let static_shape = (2..)
        .take(static_idents.len())
        .map(|v| format!(",{}", v))
        .collect::<String>();
    let short_type_suffix = self_dimensionality_form
        .iter()
        .map(|x| match *x {
            true => 'S',
            false => 'D',
        })
        .intersperse('x')
        .collect::<String>();

    // Our full implementation block
    let impl_block = format!(
        "
        impl<T{}> Tensor{ndims}{self_partial_type_suffix} {{
            /// Constructs a tensor of a given shape with values sampled from a given distribution.
            /// ```
            /// use rand::distributions::{{Uniform,Standard}};
            /// use tensor_lib::Tensor{ndims}{short_type_suffix};
            /// let x = Tensor{ndims}{short_type_suffix}::<i32{static_shape}>::from_distribution(({dynamic_shape}Uniform::<i32>::from(0..10)));
            /// let y = Tensor{ndims}{short_type_suffix}::<f32{static_shape}>::from_distribution(({dynamic_shape}Standard));
            /// ```
            pub fn from_distribution<DIST:rand::distributions::Distribution<T>>(({}distribution):({}DIST)) -> Self {{
                let mut rng = rand::thread_rng();
                Self {{ 
                    {}
                    data: distribution.sample_iter(&mut rng).take({}).collect()
                }}
            }}
        }}
        ",
        static_idents
            .iter()
            .map(|v| format!(",const {}:usize", v))
            .collect::<String>(),
        dynamic_idents
            .iter()
            .map(|v| format!("{},", v))
            .collect::<String>(),
        "usize,".repeat(dynamic_idents.len()),
        dynamic_idents
            .iter()
            .map(|v| format!("{},", v))
            .collect::<String>(),
        static_idents
            .iter()
            .chain(dynamic_idents.iter())
            .intersperse(&'*')
            .collect::<String>(),
    );
    output_string.push_str(&impl_block);
}
fn index(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
) {
    let idents = self_dimensionality_form
        .iter()
        .enumerate()
        .map(|(v, x)| {
            if *x {
                (uppercase(v), *x)
            } else {
                (lowercase(v), *x)
            }
        })
        .collect::<Vec<_>>();
    let (static_idents, _) = idents.iter().cloned().partition::<Vec<_>, _>(|(_, x)| *x);
    let idents = idents
        .into_iter()
        .map(|(v, x)| {
            if x {
                v.to_string()
            } else {
                format!("self.{}", v)
            }
        })
        .collect::<Vec<_>>();
    let static_idents = static_idents
        .into_iter()
        .map(|(v, _)| format!(",const {}:usize", v))
        .collect::<String>();

    let checks = (0..ndims)
        .zip(idents.iter())
        .map(|(i, x)| {
            format!("assert!(arr[{i}]<{x},\"Given index on dimension {i} is out of bounds.\");\n")
        })
        .collect::<String>();
    let index_sum = (0..ndims)
        .map(|i| {
            format!(
                "arr[{i}]{}",
                idents
                    .iter()
                    .rev()
                    .skip(ndims - i)
                    .map(|v| format!("*{}", v))
                    .collect::<String>()
            )
        })
        .intersperse(String::from("+"))
        .collect::<String>();
    // eprintln!("index_sum: {}",index_sum);

    // Our full implementation block
    let impl_block = format!(
        "
        impl<T:std::fmt::Debug{static_idents}> std::ops::Index<[usize;{ndims}]> for Tensor{ndims}{self_partial_type_suffix} {{
            type Output = T;
            fn index(&self,arr: [usize;{ndims}]) -> &Self::Output {{
                {checks}
                &self.data[{index_sum}]
            }}
        }}
        "
    );
    output_string.push_str(&impl_block);
}
/// Shared functionality for some similar operations (e.g. `std::ops::SubAssign`, `std::ops::AddAssign`).
fn assign_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
    op: &str,
    op_trait: &str,
    op_fn: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = bool_vec(impl_form);
        // Defines whether each dimension in our output is static or not
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { SDST } else { SDDT })
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
        // Type definition for `rhs`
        let rhs_definition = format!("{}<T{}>", impl_static_dimensions, rhs_const_generics);

        let dimension_length_checks = (0..26).take(ndims).filter_map(|d|
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
        ).collect::<String>();

        // Our full implementation block
        let impl_block = format!(
            "
            impl<T: Default + Copy + std::ops::{op_trait}{joined_const_generics}> std::ops::{op_trait}<Tensor{ndims}{rhs_definition}> for Tensor{ndims}{} {{
                fn {op_fn}(&mut self, rhs: Tensor{ndims}{rhs_definition}) {{
                    {dimension_length_checks}
                    for (a,b) in self.data.iter_mut().zip(rhs.data.iter()) {{
                        *a {op} *b;
                    }}
                }}
            }}
            ",
            // Type definition for `self`
            self_partial_type_suffix,
        );
        output_string.push_str(&impl_block);
    }
}

/// Shared functionality for some similar operations (e.g. `std::ops::Sub`, `std::ops::Add`).
fn standard_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    self_partial_type_suffix: &str,
    op: &str,
    op_trait: &str,
    op_fn: &str,
) {
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = bool_vec(impl_form);
        // Defines whether each dimension in our output is static or not
        let join = self_dimensionality_form
            .iter()
            .zip(impl_form.iter())
            .map(|(a, b)| *a || *b)
            .collect::<Vec<_>>();
        // Getting the suffix for our rhs tensor based off whether each dimensions is static or not
        let impl_static_dimensions = impl_form
            .iter()
            .map(|x| if *x { SDST } else { SDDT })
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
                .map(|x| if *x { SDST } else { SDDT })
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
/// Slicing implementations
fn slice_impl(
    output_string: &mut String,
    // The number of dimensions of the tensors involved in the operation.
    ndims: usize,
    // A slice of length `ndims` defining whether each dimension is static (`true`) or dynamic (`false`).
    self_dimensionality_form: &[bool],
    static_dimensions: &str,
    const_generics: &str,
) {
    // Iterates over all permutations of dynamic/static dimensions.
    for impl_form in (0..ndims).map(|_| (0..2)).multi_cartesian_product() {
        let impl_form = bool_vec(impl_form);

        // output_string.push_str(&impl_block);
        let const_dims = self_dimensionality_form
            .iter()
            .enumerate()
            // Filter for only the static dimensions
            .filter(|(_, b)| **b)
            .map(|(v, _)| format!(",{}", uppercase(v)))
            .collect::<String>();
        let const_slice_ranges = impl_form
            .iter()
            .enumerate()
            // Filter for only the static dimensions
            .filter(|(_, b)| **b)
            .map(|(v, _)| {
                format!(
                    "const {}{RANGE_SUFFIX}: std::ops::Range<usize>",
                    uppercase(v)
                )
            })
            .intersperse(String::from(","))
            .collect::<String>();
        let dynamic_slice_ranges = impl_form
            .iter()
            .enumerate()
            // Filter for only the dynamic dimensions
            .filter(|(_, b)| !**b)
            .map(|(v, _)| format!(",{}{RANGE_SUFFIX}: std::ops::Range<usize>", lowercase(v)))
            .collect::<String>();
        let const_slice_sizes = impl_form
            .iter()
            .enumerate()
            // Filter for only the static dimensions
            .filter(|(_, b)| **b)
            .map(|(v, _)| format!(",{{range_len({}{RANGE_SUFFIX})}}", uppercase(v)))
            .collect::<String>();
        // The contiguous lengths in our underlying vec of 1 unit along each dimension.
        let idents = self_dimensionality_form
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if *b {
                    uppercase(i).to_string()
                } else {
                    format!("self.{}", lowercase(i))
                }
            })
            .collect::<Vec<_>>();
        let range_idents = impl_form
            .iter()
            .enumerate()
            .map(|(i, b)| if *b { uppercase(i) } else { lowercase(i) })
            .collect::<Vec<_>>();

        let mut iter_str_start = String::new();
        let mut iter_str_end = String::new();
        for (i, ident) in range_idents.iter().enumerate() {
            let mut size = idents
                .iter()
                .skip(i + 1)
                .cloned()
                .intersperse(String::from("*"))
                .collect::<String>();
            if size.is_empty() {
                size = String::from("1");
            }
            let chunk_ident = lowercase(i);
            iter_str_start.push_str(&format!(
                "
                .chunks_exact({size})
                .skip({ident}{RANGE_SUFFIX}.start)
                .take({ident}{RANGE_SUFFIX}.end-{ident}{RANGE_SUFFIX}.start)
                .flat_map(|{chunk_ident}|{chunk_ident}"
            ));
            iter_str_end.push(')');
        }
        let iter_str = format!("self.data{iter_str_start}{iter_str_end}\n.collect::<Vec<_>>()");

        let dynamic_idents = impl_form
            .iter()
            .zip(range_idents.iter())
            .enumerate()
            // Filter to dynamic components
            .filter(|(_, (to, _))| !**to)
            .map(|(v, (_, from))| {
                let to = lowercase(v);
                format!(",{to}:{from}{RANGE_SUFFIX}.end-{from}{RANGE_SUFFIX}.start")
            })
            .collect::<String>();

        let impl_static_dimensions = impl_form
            .iter()
            .map(|b| if *b { SDST } else { SDDT })
            .intersperse(SDS)
            .collect::<String>();

        let slice_impl = format!(
            "
            impl<T{const_generics}> Slice{ndims}{impl_static_dimensions}<T> for Tensor{ndims}{static_dimensions}<T{const_dims}> {{
                fn slice<{const_slice_ranges}>(&self{dynamic_slice_ranges}) -> Tensor{ndims}{impl_static_dimensions}<&T{const_slice_sizes}> {{
                    Tensor{ndims}{impl_static_dimensions}::<&T{const_slice_sizes}> {{
                        data: {iter_str}
                        {dynamic_idents}
                    }}
                }}
            }}
            "
        );
        // println!("slice_impl: {}",slice_impl);
        output_string.push_str(&slice_impl);
    }
}
