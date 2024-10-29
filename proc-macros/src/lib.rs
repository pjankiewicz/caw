use heck::ToUpperCamelCase;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use std::{collections::HashMap, mem};
use syn::{DeriveInput, Data, Fields, parse_macro_input, parse_quote, punctuated::Punctuated, token::{Comma, Plus}, AngleBracketedGenericArguments, Attribute, Expr, ExprLit, GenericArgument, GenericParam, Generics, Ident, ItemStruct, Lit, LitStr, Meta, Path, PathArguments, PathSegment, Token, Type, TypeParam, TypeParamBound, TypePath, WhereClause, WherePredicate, DataStruct, Field};
use syn::spanned::Spanned;

fn convert_snake_to_camel(ident: Ident) -> Ident {
    let upper_camel_case_string =
        ident.to_string().as_str().to_upper_camel_case();
    Ident::new(upper_camel_case_string.as_str(), ident.span())
}

fn generic_type_param(
    ident: Ident,
    bounds: &Punctuated<TypeParamBound, Plus>,
) -> TypeParam {
    TypeParam {
        attrs: Vec::new(),
        ident,
        colon_token: None,
        bounds: bounds.clone(),
        eq_token: None,
        default: None,
    }
}

fn type_of_single_segment(path_segment: PathSegment) -> Type {
    let mut segments = Punctuated::new();
    segments.push(path_segment);
    Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: None,
            segments,
        },
    })
}

/// A type represented by a single identifier.
fn type_of_ident(ident: Ident) -> Type {
    type_of_single_segment(PathSegment {
        ident,
        arguments: PathArguments::None,
    })
}

fn bracketed_generic_arguments(
    args: Punctuated<GenericArgument, Comma>,
) -> AngleBracketedGenericArguments {
    AngleBracketedGenericArguments {
        colon2_token: None,
        lt_token: Token![<](Span::call_site()),
        args,
        gt_token: Token![>](Span::call_site()),
    }
}

fn type_with_type_params(
    ident: Ident,
    args: Punctuated<GenericArgument, Comma>,
) -> Type {
    let generic_arguments = bracketed_generic_arguments(args);
    let path_segment = PathSegment {
        ident,
        arguments: PathArguments::AngleBracketed(generic_arguments),
    };
    type_of_single_segment(path_segment)
}

fn generics_for_function_def(
    params: Punctuated<GenericParam, Comma>,
) -> Generics {
    Generics {
        lt_token: Some(Token![<](Span::call_site())),
        params,
        gt_token: Some(Token![>](Span::call_site())),
        where_clause: None,
    }
}

fn attr_lit_str(attr: &Attribute) -> Option<&LitStr> {
    if let Meta::NameValue(ref meta_name_value) = attr.meta {
        if let Expr::Lit(ExprLit {
            lit: Lit::Str(ref s),
            ..
        }) = meta_name_value.value
        {
            return Some(s);
        }
    }
    None
}

/// Pass this a struct definition to generate a constructor, setters,
/// and a build method treating the struct as a builder in the builder
/// pattern.
#[proc_macro]
pub fn builder(input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as ItemStruct);
    let builder_ident = input.ident.clone();
    let mut all_field_idents_in_order: Punctuated<Ident, Comma> =
        Punctuated::new();
    let mut field_without_default_idents = Vec::new();
    let mut field_without_default_types = Vec::new();
    let mut field_with_default_idents = Vec::new();
    let mut field_with_default_types = Vec::new();
    let mut regular_field_with_default_idents = Vec::new();
    let mut regular_field_with_default_types = Vec::new();
    let mut generic_field_with_default_idents = Vec::new();
    let mut generic_field_with_default_types = Vec::new();
    let mut generic_field_with_default_constraints = Vec::new();
    let mut generic_field_with_default_extra_where_predicates = Vec::new();
    let mut field_default_values = Vec::new();
    let mut generic_field_with_default_type_param_idents = Vec::new();
    let mut generic_field_type_to_default_type = HashMap::new();
    let mut build_fn = Vec::new();
    let mut build_ty = Vec::new();
    let mut constructor = None;
    let mut constructor_generics: Punctuated<GenericParam, Comma> =
        Punctuated::new();
    let mut generic_setter_type_name = None;
    let mut constructor_doc = None;
    let mut constructor_where_predicates = Vec::new();
    let attrs = mem::replace(&mut input.attrs, Vec::new());
    for attr in attrs {
        if attr.path().is_ident("constructor") {
            if let Some(s) = attr_lit_str(&attr) {
                let ident: Ident = s.parse().expect("Expected identifier");
                constructor = Some(ident);
            }
            continue;
        }
        if attr.path().is_ident("constructor_doc") {
            if let Some(s) = attr_lit_str(&attr) {
                constructor_doc = Some(s.value());
            }
            continue;
        }
        if attr.path().is_ident("constructor_where") {
            if let Some(s) = attr_lit_str(&attr) {
                let where_predicates: WherePredicate =
                    s.parse().expect("Expected where predicates");
                constructor_where_predicates.push(where_predicates);
            }
            continue;
        }
        if attr.path().is_ident("build_fn") {
            if let Some(s) = attr_lit_str(&attr) {
                let ident: Path = s.parse().expect("Expected identifier");
                build_fn.push(ident);
            }
            continue;
        }
        if attr.path().is_ident("build_ty") {
            if let Some(s) = attr_lit_str(&attr) {
                let ty: Type = s.parse().expect("Expected type");
                build_ty.push(ty);
            }
            continue;
        }
        if attr.path().is_ident("generic_setter_type_name") {
            if let Some(s) = attr_lit_str(&attr) {
                let ident: Ident = s.parse().expect("Expected identifier");
                generic_setter_type_name = Some(ident);
            }
            continue;
        }
        input.attrs.push(attr);
    }
    let generic_setter_type_name = generic_setter_type_name
        .unwrap_or_else(|| Ident::new("__T", Span::call_site()));
    let constructor = constructor.expect("Missing \"constructor\" attribute");
    let constructor_doc = format!(
        " {}",
        constructor_doc.unwrap_or_else(|| format!(
            "Shorthand for [`{}::new`].",
            builder_ident
        ))
    );
    let constructor_doc: Attribute = parse_quote!(#[doc = #constructor_doc]);
    let constructor_where_clause = WhereClause {
        where_token: Token![where](Span::call_site()),
        predicates: {
            let mut predicates = Punctuated::new();
            for where_predicate in constructor_where_predicates {
                predicates.push(where_predicate);
            }
            predicates
        },
    };
    if build_fn.len() != build_ty.len() || build_fn.len() > 1 {
        panic!(
            "The `build_fn` and `build_ty` attributes should both be set \
            exactly once, or not set at all."
        );
    }
    for field in input.fields.iter_mut() {
        if let Some(ident) = field.ident.as_ref() {
            all_field_idents_in_order.push(ident.clone());
            let mut default = None;
            let mut generic = false;
            let mut generic_with_constraints: Punctuated<TypeParamBound, Plus> =
                Punctuated::new();
            let mut extra_where_predicates: Punctuated<WherePredicate, Comma> =
                Punctuated::new();
            let mut generic_name = None;
            let attrs = mem::replace(&mut field.attrs, Vec::new());
            for attr in attrs {
                if attr.path().is_ident("default") {
                    if let Meta::NameValue(ref meta_name_value) = attr.meta {
                        default = Some(meta_name_value.value.clone());
                        continue;
                    }
                }
                if attr.path().is_ident("generic") {
                    generic = true;
                    continue;
                }
                if attr.path().is_ident("generic_with_constraint") {
                    generic = true;
                    if let Some(s) = attr_lit_str(&attr) {
                        let constraint: TypeParamBound =
                            s.parse().expect("Expected constraint");
                        generic_with_constraints.push(constraint);
                    }
                    continue;
                }
                if attr.path().is_ident("generic_name") {
                    generic = true;
                    if let Some(s) = attr_lit_str(&attr) {
                        let name: Ident =
                            s.parse().expect("Expected identifier");
                        generic_name = Some(name);
                    }
                    continue;
                }
                if attr.path().is_ident("extra_where_predicate") {
                    generic = true;
                    if let Some(s) = attr_lit_str(&attr) {
                        let where_predicate: WherePredicate =
                            s.parse().expect("Expected where predicates");
                        extra_where_predicates.push(where_predicate);
                    }
                    continue;
                }
                field.attrs.push(attr);
            }
            if generic {
                let generic_ident = generic_name.unwrap_or_else(|| {
                    format_ident!(
                        "__{}T",
                        convert_snake_to_camel(ident.clone())
                    )
                });
                let generic_type_param = generic_type_param(
                    generic_ident,
                    &generic_with_constraints,
                );
                if default.is_some() {
                    generic_field_type_to_default_type.insert(
                        generic_type_param.ident.clone(),
                        field.ty.clone(),
                    );
                    generic_field_with_default_type_param_idents
                        .push(generic_type_param.ident.clone());
                } else {
                    constructor_generics
                        .push(GenericParam::Type(generic_type_param.clone()));
                }
                field.ty = type_of_ident(generic_type_param.ident.clone());
                input
                    .generics
                    .params
                    .push(GenericParam::Type(generic_type_param));
            }
            if let Some(default) = default {
                field_with_default_idents.push(ident.clone());
                field_with_default_types.push(field.ty.clone());
                if generic {
                    generic_field_with_default_idents.push(ident.clone());
                    generic_field_with_default_types.push(field.ty.clone());
                    generic_field_with_default_constraints
                        .push(generic_with_constraints);
                    generic_field_with_default_extra_where_predicates
                        .push(extra_where_predicates);
                } else {
                    regular_field_with_default_idents.push(ident.clone());
                    regular_field_with_default_types.push(field.ty.clone());
                }
                field_default_values.push(default);
            } else {
                field_without_default_idents.push(ident.clone());
                field_without_default_types.push(field.ty.clone());
            }
        }
    }
    let constructor_generics = generics_for_function_def(constructor_generics);
    let new_fn_return_type_generics = {
        let mut args = Punctuated::new();
        for generic_param in &input.generics.params {
            if let GenericParam::Type(ref type_param) = generic_param {
                let ty = if let Some(default_type) =
                    generic_field_type_to_default_type.get(&type_param.ident)
                {
                    // This type param has a default value, so the
                    // abstract type parameter name won't match the
                    // type of the field. Use the type from the
                    // original struct definition as the default type
                    // of the field.
                    default_type.clone()
                } else {
                    // No default value, so `new` will be passed a
                    // value. This means we can use the name of the
                    // type parameter as this type argument. This case
                    // is also hit for any non-generic type
                    // parameters.
                    type_of_ident(type_param.ident.clone())
                };
                args.push(GenericArgument::Type(ty));
            }
        }
        bracketed_generic_arguments(args)
    };
    // The return type of the `Builder::new` method.
    let new_fn_return_type = {
        let path_segment = PathSegment {
            ident: builder_ident.clone(),
            arguments: PathArguments::AngleBracketed(
                new_fn_return_type_generics.clone(),
            ),
        };
        type_of_single_segment(path_segment)
    };
    let make_setter_return_type = |type_param_ident| {
        let mut args = Punctuated::new();
        for generic_param in &input.generics.params {
            if let GenericParam::Type(ref type_param) = generic_param {
                let type_ident = if &type_param.ident == type_param_ident {
                    // The current type parameter is the one that
                    // should be replaced by the argument type of
                    // the getter of the current field.
                    generic_setter_type_name.clone()
                } else {
                    type_param.ident.clone()
                };
                args.push(GenericArgument::Type(type_of_ident(type_ident)));
            }
        }
        type_with_type_params(builder_ident.clone(), args)
    };
    let generic_field_with_default_setter_return_types =
        generic_field_with_default_type_param_idents
            .iter()
            .map(make_setter_return_type)
            .collect::<Vec<_>>();
    let all_fields_except_current = |current_field_ident: &Ident| {
        let mut other_fields: Punctuated<Ident, Comma> = Punctuated::new();
        for field in input.fields.iter() {
            if let Some(ref field_ident) = field.ident {
                if field_ident != current_field_ident {
                    other_fields.push(field_ident.clone());
                }
            }
        }
        other_fields
    };
    let regular_field_with_default_setter_all_fields_except_current =
        regular_field_with_default_idents
            .iter()
            .map(all_fields_except_current)
            .collect::<Vec<_>>();
    let generic_field_with_default_setter_all_fields_except_current =
        generic_field_with_default_idents
            .iter()
            .map(all_fields_except_current)
            .collect::<Vec<_>>();
    let (impl_generics, ty_generics, where_clause) =
        input.generics.split_for_impl();
    let expanded = quote! {
        #input

        impl #impl_generics #builder_ident #ty_generics #where_clause {

            // Create a new builder with default values set for some
            // fields, and with other fields set by arguments to this
            // method. Note that the return type is not `Self`, as the
            // type parameters of fields with default values are
            // concrete (whatever the type of the default value for
            // the field is) raher than abstract.
            pub fn new(
                #(#field_without_default_idents: #field_without_default_types),*
            ) -> #new_fn_return_type {
                #builder_ident {
                    #(#field_without_default_idents),*,
                    #(#field_with_default_idents: #field_default_values),*,
                }
            }

            // Generate a setter function for each regular field with
            // a default value. Fields without default values are set
            // in the `new` function instead.
            #(pub fn #regular_field_with_default_idents(
                    self,
                    #regular_field_with_default_idents: #regular_field_with_default_types,
            ) -> Self {
                let Self {
                    #regular_field_with_default_setter_all_fields_except_current,
                    ..
                } = self;
                #builder_ident {
                    #regular_field_with_default_setter_all_fields_except_current,
                    #regular_field_with_default_idents,
                }
            })*

            // Generate a setter function for each generic field with
            // a default value. Fields without default values are set
            // in the `new` function instead.
            #(pub fn #generic_field_with_default_idents<#generic_setter_type_name>(
                self,
                #generic_field_with_default_idents: #generic_setter_type_name,
            ) -> #generic_field_with_default_setter_return_types
            where
                #generic_setter_type_name: #generic_field_with_default_constraints,
                #generic_field_with_default_extra_where_predicates
            {
                let Self {
                    #generic_field_with_default_setter_all_fields_except_current,
                    ..
                } = self;
                #builder_ident {
                    #generic_field_with_default_setter_all_fields_except_current,
                    #generic_field_with_default_idents,
                }
            })*

            // Call the user-provided `build_fn` if any. If no
            // `build_fn` was set, don't generate a `build`
            // method. This is a valid use-case, as a user may wish to
            // implement the `build` method by hand for some
            // non-trivial builders.
            #(pub fn build(self) -> #build_ty {
                let Self { #all_field_idents_in_order } = self;
                #build_fn(#all_field_idents_in_order)
            })*
        }

        #constructor_doc
        pub fn #constructor #constructor_generics (
            #(#field_without_default_idents: #field_without_default_types),*
        ) -> #new_fn_return_type
            #constructor_where_clause
        {
            #builder_ident::#new_fn_return_type_generics::new(#(#field_without_default_idents),*)
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(FromValue)]
pub fn derive_from_value(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    // Get the name of the enum
    let name = &input.ident;
    let variants = match &input.data {
        Data::Enum(data_enum) => &data_enum.variants,
        _ => panic!("FromValue can only be derived for enums"),
    };

    // Generate `impl From` for each variant
    let from_impls = variants.iter().filter_map(|variant| {
        let variant_ident = &variant.ident;
        match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let ty = &fields.unnamed.first().unwrap().ty;
                Some(quote! {
	                    impl From<#ty> for #name {
	                        fn from(value: #ty) -> Self {
	                            #name::#variant_ident(value)
	                        }
	                    }
	                })
            }
            _ => None,
        }
    });

    let expanded = quote! {
	        #(#from_impls)*
	    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(FromExpr, attributes(init, optional, skip, from_expr))]
pub fn derive_from_expr(input: TokenStream) -> TokenStream {
    // Parse the input struct
    let input = parse_macro_input!(input as DeriveInput);

    // Generate the code
    let gen = impl_from_expr(&input);

    // Return the generated code
    TokenStream::from(gen)
}
fn impl_from_expr(input: &DeriveInput) -> proc_macro2::TokenStream {
    let struct_name = &input.ident;

    // Extract the variant name from attributes or default to struct name
    let variant_name = get_variant_name(&input.attrs, struct_name);

    // Extract the fields
    let fields = if let Data::Struct(DataStruct {
                                         fields: Fields::Named(ref fields),
                                         ..
                                     }) = input.data
    {
        fields.named.iter().collect::<Vec<&Field>>()
    } else {
        panic!("FromExpr can only be derived for structs with named fields");
    };

    // Initialize vectors for different types of fields
    let mut init_fields = Vec::new();
    let mut optional_fields = Vec::new();
    let mut skipped_fields = Vec::new();
    let mut all_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();

        all_fields.push(field_name);

        let mut is_init = false;
        let mut is_optional = false;
        let mut is_skip = false;

        // Check attributes
        for attr in &field.attrs {
            if attr.path().is_ident("init") {
                is_init = true;
            } else if attr.path().is_ident("optional") {
                is_optional = true;
            } else if attr.path().is_ident("skip") {
                is_skip = true;
            }
        }

        if is_skip {
            skipped_fields.push(field);
            continue;
        }

        if is_init {
            init_fields.push(field);
        } else if is_optional {
            optional_fields.push(field);
        } else {
            // Detect Option types
            if is_option_type(&field.ty) {
                optional_fields.push(field);
            } else {
                init_fields.push(field);
            }
        }
    }

    // Build the code for required parameters
    let mut required_params = Vec::new();
    let mut required_conversions = Vec::new();
    let mut required_expr_fields = Vec::new();
    let mut all_expr_fields = Vec::new();

    for field in all_fields {
        all_expr_fields.push(quote! { #field });
    }

    for field in init_fields {
        let field_name = field.ident.as_ref().unwrap();
        required_params.push(quote! { #field_name });
        required_expr_fields.push(quote! { #field_name });

        let field_type_ident = get_base_type(&field.ty);

        required_conversions.push(quote! {
            let #field_name: #field_type_ident = #field_name.eval()?.try_into()
                .map_err(|_| anyhow::anyhow!("Could not convert to {}", stringify!(#field_type_ident)))?;
        });
    }

    // Build the code for optional parameters
    let mut optional_conversions = Vec::new();
    let mut optional_expr_fields = Vec::new();

    for field in optional_fields {
        let field_name = field.ident.as_ref().unwrap();
        optional_expr_fields.push(quote! { #field_name });

        let base_type = get_option_inner_type(&field.ty).unwrap_or_else(|| field.ty.clone());
        let field_type_ident = get_base_type(&base_type);

        optional_conversions.push(quote! {
            if let Some(#field_name) = #field_name {
                let #field_name: #field_type_ident = #field_name.eval()?.try_into()
                    .map_err(|_| anyhow::anyhow!("Could not convert to {}", stringify!(#field_type_ident)))?;
                builder = builder.#field_name(#field_name);
            }
        });
    }

    // Construct the builder initialization
    let builder_new_call = if required_params.is_empty() {
        quote! { let mut builder = Self::new(); }
    } else {
        quote! { let mut builder = Self::new(#(#required_params),*); }
    };

    // Combine everything into the final method
    let method = quote! {
        impl crate::ast::expr::FromExpr for #struct_name {
            fn from_expr(expr: &crate::ast::expr::Expr) -> Result<crate::ast::value::Value, anyhow::Error> {
                use anyhow::anyhow;

                if let crate::ast::expr::Expr::#variant_name { #(#all_expr_fields),*, } = expr {
                    #(#required_conversions)*

                    #builder_new_call

                    #(#optional_conversions)*

                    Ok(builder.build().into())
                } else {
                    Err(anyhow!("Invalid expression for {}", stringify!(#struct_name)))
                }
            }
        }
    };

    method
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty {
        path.segments
            .last()
            .map_or(false, |seg| seg.ident == "Option")
    } else {
        false
    }
}

fn get_option_inner_type(ty: &Type) -> Option<Type> {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(segment) = path.segments.last() {
            if segment.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_type)) = args.args.first() {
                        return Some(inner_type.clone());
                    }
                }
            }
        }
    }
    None
}

fn get_base_type(ty: &Type) -> Ident {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(segment) = path.segments.last() {
            return segment.ident.clone();
        }
    }
    panic!("Unsupported type");
}

fn get_variant_name(attrs: &[Attribute], struct_name: &Ident) -> Ident {
    for attr in attrs {
        if attr.path().is_ident("from_expr") {
            let args = attr.parse_args::<FromExprArgs>();
            if let Ok(args) = args {
                return args.variant_name;
            }
        }
    }
    struct_name.clone()
}

// Struct to parse the attribute arguments
struct FromExprArgs {
    variant_name: Ident,
}

impl syn::parse::Parse for FromExprArgs {
    fn parse(input: syn::parse::ParseStream) -> Result<Self, syn::Error> {
        let ident: Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;

        if ident == "variant_name" {
            let lit_str: LitStr = input.parse()?;
            let variant_name = Ident::new(&lit_str.value(), lit_str.span());
            Ok(FromExprArgs { variant_name })
        } else {
            Err(syn::Error::new(ident.span(), "Expected 'variant_name'"))
        }
    }
}

// CGP
#[proc_macro_derive(FunctionSet, attributes(exclude_from_cgp))]
pub fn derive_function_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let enum_name = &input.ident;

    // Ensure we're deriving on an enum
    let enum_data = match input.data {
        Data::Enum(ref data_enum) => data_enum,
        _ => panic!("FunctionSet can only be derived on enums"),
    };

    let mut function_variants = Vec::new();

    // Iterate over each variant
    let mut idx = 0 as usize;
    for (_, variant) in enum_data.variants.iter().enumerate() {
        // Check for the exclude attribute
        let exclude = variant
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("exclude_from_cgp"));

        if exclude {
            continue;
        }

        // Get the variant name
        let variant_name = &variant.ident;

        // Calculate arity and generate field assignments
        let (arity, field_assignments) = calculate_arity_and_assignments(&variant.fields);

        function_variants.push((idx, variant_name.clone(), arity, field_assignments));

        idx += 1;
    }

    // Generate the function_set method
    let function_set_entries = function_variants.iter().map(
        |(idx, variant_name, arity, field_assignments)| {
            let arity_value = match arity {
                Some(n) => quote! { Some(#n) },
                None => quote! { None }, // Variable arity
            };

            if field_assignments.is_empty() {
                quote! {
                    {
                        fn constructor(inputs: &[Expr], input_idx: &mut usize) -> Expr {
                            Expr::#variant_name
                        }
                        (stringify!(#variant_name).to_string(), #idx, #arity_value, constructor)
                    }
                }
            } else {
                quote! {
                    {
                        fn constructor(inputs: &[Expr], input_idx: &mut usize) -> Expr {
                            Expr::#variant_name {
                                #(#field_assignments)*
                            }
                        }
                        (stringify!(#variant_name).to_string(), #idx, #arity_value, constructor)
                    }
                }
            }
        },
    );

    let expanded = quote! {
        impl Expr {
            pub fn function_set() -> Vec<(String, usize, Option<usize>, fn(&[Expr], &mut usize) -> Expr)> {
                vec![
                    #(#function_set_entries),*
                ]
            }
        }
    };

    TokenStream::from(expanded)
}

fn calculate_arity_and_assignments(
    fields: &Fields,
) -> (Option<usize>, Vec<proc_macro2::TokenStream>) {
    let mut arity = 0;
    let mut has_variable = false;
    let mut assignments = Vec::new();

    match fields {
        Fields::Named(fields_named) => {
            for field in fields_named.named.iter() {
                let field_name = field.ident.clone().unwrap();
                let ty = &field.ty;
                let (field_arity, assignment) = generate_field_assignment(&field_name, ty);
                if field_arity.is_none() {
                    has_variable = true;
                } else {
                    arity += field_arity.unwrap();
                }
                assignments.push(assignment);
            }
        }
        Fields::Unnamed(fields_unnamed) => {
            for (idx, field) in fields_unnamed.unnamed.iter().enumerate() {
                let field_name = Ident::new(&format!("field{}", idx), field.span());
                let ty = &field.ty;
                let (field_arity, assignment) = generate_field_assignment(&field_name, ty);
                if field_arity.is_none() {
                    has_variable = true;
                } else {
                    arity += field_arity.unwrap();
                }
                assignments.push(assignment);
            }
        }
        Fields::Unit => {}
    }

    let total_arity = if has_variable { None } else { Some(arity) };
    (total_arity, assignments)
}

fn generate_field_assignment(
    field_name: &Ident,
    ty: &Type,
) -> (Option<usize>, proc_macro2::TokenStream) {
    if is_box_expr(ty) {
        // Box<Expr> field
        let assignment = quote! {
            #field_name: {
                let expr = Box::new(inputs[*input_idx].clone());
                *input_idx += 1;
                expr
            },
        };
        (Some(1), assignment)
    } else if is_option_box_expr(ty) {
        // Option<Box<Expr>> field
        let assignment = quote! {
            #field_name: {
                let expr = Some(Box::new(inputs[*input_idx].clone()));
                *input_idx += 1;
                expr
            },
        };
        (Some(1), assignment)
    } else if is_vec_box_expr(ty) {
        // Vec<Box<Expr>> field
        let assignment = quote! {
            #field_name: {
                let remaining_inputs = &inputs[*input_idx..];
                let vec = remaining_inputs.iter().map(|e| Box::new(e.clone())).collect();
                *input_idx = inputs.len();
                vec
            },
        };
        (None, assignment) // Variable arity
    } else {
        // Other types, provide a default value or handle as needed
        let assignment = quote! {
            #field_name: Default::default(),
        };
        (Some(0), assignment)
    }
}

// Helper functions to identify field types
fn is_box_expr(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Box" {
            if let syn::PathArguments::AngleBracketed(angle_bracketed) = &segments[0].arguments {
                if angle_bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(inner_ty) = &angle_bracketed.args[0] {
                        if is_expr_type(inner_ty) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

fn is_option_box_expr(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Option" {
            if let syn::PathArguments::AngleBracketed(angle_bracketed) = &segments[0].arguments {
                if angle_bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(inner_ty) = &angle_bracketed.args[0] {
                        return is_box_expr(inner_ty);
                    }
                }
            }
        }
    }
    false
}

fn is_vec_box_expr(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Vec" {
            if let syn::PathArguments::AngleBracketed(angle_bracketed) = &segments[0].arguments {
                if angle_bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(inner_ty) = &angle_bracketed.args[0] {
                        return is_box_expr(inner_ty);
                    }
                }
            }
        }
    }
    false
}

fn is_expr_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        let segments = &type_path.path.segments;
        if segments.len() == 1 && segments[0].ident == "Expr" {
            return true;
        }
    }
    false
}
