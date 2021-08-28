extern crate proc_macro;

use proc_macro2::TokenStream;
use quote::quote;

const LIB: LibIdent = LibIdent;
struct LibIdent;
impl quote::ToTokens for LibIdent {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote! { gpukit });
    }
}

const WGPU: WgpuIdent = WgpuIdent;
struct WgpuIdent;
impl quote::ToTokens for WgpuIdent {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(quote! { gpukit::wgpu });
    }
}

#[proc_macro_derive(Bindings, attributes(uniform, texture, sampler))]
pub fn derive_bindings(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);

    match derive_bindings_impl(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn combine_errors(errors: impl IntoIterator<Item = syn::Error>) -> syn::Result<()> {
    let mut errors = errors.into_iter();

    if let Some(mut base_error) = errors.next() {
        for error in errors {
            base_error.combine(error);
        }

        Err(base_error)
    } else {
        Ok(())
    }
}

fn derive_bindings_impl(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    match &input.data {
        syn::Data::Enum(_) | syn::Data::Union(_) => Err(syn::Error::new(
            input.ident.span(),
            "`Bindings` may only be derived for `struct`s",
        )),
        syn::Data::Struct(struc) => {
            let mut errors = Vec::new();

            let mut layout_entries = Vec::new();
            let mut binding_entries = Vec::new();

            for (field_index, field) in struc.fields.iter().enumerate() {
                let field_ident = &field
                    .ident
                    .as_ref()
                    .map(|ident| quote! { #ident })
                    .unwrap_or_else(|| quote! { #field_index });

                match struct_field_binding(field, quote! { self.#field_ident }) {
                    Ok(binding) => {
                        layout_entries.push(binding.layout);
                        binding_entries.push(binding.entry);
                    }
                    Err(error) => errors.push(error),
                }
            }

            combine_errors(errors)?;

            Ok(create_struct_impl(
                &input,
                layout_entries.into_iter(),
                binding_entries.into_iter(),
            ))
        }
    }
}

struct Binding {
    layout: TokenStream,
    entry: TokenStream,
}

struct BindingDesc {
    binding: syn::Expr,
    stage: Option<syn::Expr>,
    binding_type: TokenStream,
    resource: TokenStream,
}

fn struct_field_binding(field: &syn::Field, accessor: TokenStream) -> syn::Result<Binding> {
    let mut attributes = field.attrs.iter().filter_map(BindingAttribute::parse);

    match (attributes.next(), attributes.next()) {
        (None, _) => Err(syn::Error::new_spanned(
            field,
            "must specify binding type using `#[uniform(...)]`, ...",
        )),
        (Some(_), Some(_)) => Err(syn::Error::new_spanned(
            field,
            "may only specify a single binding type using `#[uniform(...)]`, ...",
        )),

        (Some(attribute), None) => match attribute {
            Err(error) => Err(error),
            Ok(attribute) => {
                let field_type = &field.ty;

                let desc = match attribute {
                    BindingAttribute::Uniform(UniformAttribute { binding, stage }) => BindingDesc {
                        binding,
                        stage,
                        binding_type: quote! {
                            #WGPU::BindingType::Buffer {
                                ty: #WGPU::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: <#field_type as #LIB::BufferResource>::ELEMENT_SIZE,
                            }
                        },
                        resource: quote! {
                            #WGPU::BindingResource::Buffer(
                                <#field_type as #LIB::BufferResource>::buffer_binding(&#accessor)
                            )
                        },
                    },
                    BindingAttribute::Texture(TextureAttribute { binding, stage, filterable }) => BindingDesc {
                        binding,
                        stage,
                        binding_type: quote! {
                            #WGPU::BindingType::Texture {
                                sample_type: #WGPU::TextureSampleType::Float { filterable: #filterable },
                                view_dimension: #WGPU::TextureViewDimension::D2,
                                multisampled: false,
                            }
                        },
                        resource: quote! {
                            #WGPU::BindingResource::TextureView(
                                <#field_type as #LIB::TextureResource>::texture_view_binding(&#accessor)
                            )
                        },
                    },
                    BindingAttribute::Sampler(SamplerAttribute { binding, stage, filtering }) => BindingDesc {
                        binding,
                        stage,
                        binding_type: quote! {
                            #WGPU::BindingType::Sampler {
                                filtering: #filtering,
                                comparison: false,
                            }
                        },
                        resource: quote! {
                            #WGPU::BindingResource::Sampler(&#accessor)
                        },
                    },
                };

                Ok(Binding::from_desc(desc))
            }
        },
    }
}

impl Binding {
    fn from_desc(desc: BindingDesc) -> Binding {
        let BindingDesc {
            binding,
            stage,
            binding_type,
            resource,
        } = desc;

        let visibility = stage
            .map(|stage| quote! { #stage })
            .unwrap_or_else(|| quote! { #WGPU::ShaderStages::all() });

        let layout = quote! {
            #WGPU::BindGroupLayoutEntry {
                binding: #binding,
                visibility: #visibility,
                count: None,
                ty: #binding_type
            }
        };

        let entry = quote! {
            #WGPU::BindGroupEntry {
                binding: #binding,
                resource: #resource,
            }
        };

        Binding { layout, entry }
    }
}

fn create_struct_impl(
    input: &syn::DeriveInput,
    layout_entries: impl Iterator<Item = TokenStream>,
    binding_entries: impl Iterator<Item = TokenStream>,
) -> TokenStream {
    let ident = &input.ident;

    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_generics #LIB::Bindings for #ident #type_generics #where_clause {
            const LABEL: Option<&'static str> = None;

            const LAYOUT_ENTRIES: &'static [#WGPU::BindGroupLayoutEntry] = &[
                #(#layout_entries),*
            ];

            fn create_bind_group(
                &self,
                device: &#WGPU::Device,
                layout: &#WGPU::BindGroupLayout
            ) -> #WGPU::BindGroup {
                device.create_bind_group(&#WGPU::BindGroupDescriptor {
                    label: Self::LABEL,
                    layout,
                    entries: &[
                        #(#binding_entries),*
                    ]
                })
            }
        }
    }
}

#[derive(Debug)]
enum BindingAttribute {
    Uniform(UniformAttribute),
    Texture(TextureAttribute),
    Sampler(SamplerAttribute),
}

#[derive(Debug)]
struct UniformAttribute {
    binding: syn::Expr,
    stage: Option<syn::Expr>,
}

#[derive(Debug)]
struct TextureAttribute {
    binding: syn::Expr,
    stage: Option<syn::Expr>,
    filterable: bool,
}

#[derive(Debug)]
struct SamplerAttribute {
    binding: syn::Expr,
    stage: Option<syn::Expr>,
    filtering: bool,
}


impl BindingAttribute {
    fn parse(attribute: &syn::Attribute) -> Option<syn::Result<BindingAttribute>> {
        let mut items = match attribute.parse_args::<MetaList>() {
            Ok(meta) => meta.0.into_iter().collect::<Vec<MetaListItem>>(),
            Err(error) => return Some(Err(error)),
        };

        if let Err(e) = check_duplicate_meta_items(&mut items) {
            return Some(Err(e));
        }

        if attribute.path.is_ident("uniform") {
            Some(UniformAttribute::parse(items, attribute).map(BindingAttribute::Uniform))
        } else if attribute.path.is_ident("texture") {
            Some(TextureAttribute::parse(items, attribute).map(BindingAttribute::Texture))
        } else if attribute.path.is_ident("sampler") {
            Some(SamplerAttribute::parse(items, attribute).map(BindingAttribute::Sampler))
        } else {
            None
        }
    }
}

macro_rules! impl_parse_attributes {
    ($ident:ident {
        $( #[$kind:ident] $field:ident ),* $(,)?
    }) => {
        #[allow(clippy::redundant_field_names)]
        impl $ident {
            fn parse(items: Vec<MetaListItem>, attribute: &syn::Attribute) -> syn::Result<Self> {
                let mut errors = Vec::new();
                let mut emit_error = |error: syn::Error| {
                    errors.push(error);
                };

                $(
                    let mut $field = None;
                )*

                for item in items {
                    match item.key.to_string().as_str() {
                        $(
                            stringify!($field) => {
                                impl_parse_attributes!(@parse_field(#[$kind] $field, item, emit_error) )
                            },
                        )*
                        _ => emit_error(syn::Error::new(
                                item.key.span(),
                                format!(
                                    concat!("unknown binding option: `{}`", stringify!($($field),*)),
                                    item.key
                                ),
                        )),
                    }
                }

                combine_errors(errors)?;

                let error = |message: &str| {
                    syn::Error::new_spanned(attribute, message)
                };

                Ok($ident {
                    $(
                        $field: impl_parse_attributes!(@validate_field(#[$kind] $field, error) )
                    ),*
                })
            }
        }
    };

    (@parse_field(#[required] $field:ident, $item:expr, $error:ident)) => {
        match expect_expr($item) {
            Ok(__value) => $field = Some(__value),
            Err(error) => $error(error),
        }
    };
    (@validate_field(#[required] $field:ident, $error:ident)) => {
        match $field {
            Some(value) => value,
            None => return Err($error(
                concat!("missing required attribute: `", stringify!($field), " = ...`")
            )),
        }
    };

    (@parse_field(#[optional] $field:ident, $item:expr, $error:ident)) => {
        match expect_expr($item) {
            Ok(__value) => $field = Some(__value),
            Err(error) => $error(error),
        }
    };
    (@validate_field(#[optional] $field:ident, $error:ident)) => {
        $field
    };

    (@parse_field(#[flag] $field:ident, $item:expr, $error:ident)) => {
        match expect_flag($item) {
            Ok(()) => $field = Some(()),
            Err(error) => $error(error),
        }
    };
    (@validate_field(#[flag] $field:ident, $error:ident)) => {
        $field.is_some()
    };
}

impl_parse_attributes! {
    UniformAttribute {
        #[required] binding,
        #[optional] stage,
    }
}

impl_parse_attributes! {
    TextureAttribute {
        #[required] binding,
        #[optional] stage,
        #[flag] filterable,
    }
}

impl_parse_attributes! {
    SamplerAttribute {
        #[required] binding,
        #[optional] stage,
        #[flag] filtering,
    }
}


struct MetaList(syn::punctuated::Punctuated<MetaListItem, syn::Token![,]>);

struct MetaListItem {
    key: syn::Ident,
    value: Option<syn::Expr>,
}

fn check_duplicate_meta_items(items: &mut [MetaListItem]) -> syn::Result<()> {
    let mut errors = Vec::new();

    items.sort_by_cached_key(|item| item.key.to_string());
    for window in items.windows(2) {
        if let [a, b] = window {
            if a.key == b.key {
                errors.push(syn::Error::new(
                    b.key.span(),
                    format!("duplicate binding option `{}`", b.key),
                ))
            }
        }
    }

    combine_errors(errors)
}

impl syn::parse::Parse for MetaList {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let items = syn::punctuated::Punctuated::parse_terminated(input)?;
        Ok(MetaList(items))
    }
}

impl syn::parse::Parse for MetaListItem {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let key = input.parse::<syn::Ident>()?;
        let eq_token: Option<syn::Token![=]> = input.parse()?;

        let value = if eq_token.is_some() {
            Some(input.parse()?)
        } else {
            None
        };

        Ok(MetaListItem { key, value })
    }
}

fn expect_expr(item: MetaListItem) -> syn::Result<syn::Expr> {
    match item.value {
        None => Err(syn::Error::new(
            item.key.span(),
            format!("expected a value: `{} = ...`", item.key),
        )),
        Some(expr) => Ok(expr),
    }
}

fn expect_flag(item: MetaListItem) -> syn::Result<()> {
    match item.value {
        None => Ok(()),
        Some(_) => Err(syn::Error::new(
            item.key.span(),
            format!("unexpected value, expected simple flag: `{}`", item.key)
        )),
    }
}
