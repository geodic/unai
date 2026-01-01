use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::Parse, parse_macro_input, FnArg, GenericArgument, ItemImpl, LitStr, PathArguments,
    ReturnType, Token, Type,
};

/// Proc macro attribute for tools that automatically extracts Input/Output types from the `call` method.
///
/// Apply this to an `impl` block that contains an async `call` method.
/// The macro will:
/// - Use the provided tool name and description from attributes
/// - Infer Input type from the call method parameter
/// - Infer Output type from the return type Result<T, ToolError>
/// - Generate definition() method that returns ToolDefinition
/// - Generate internal execution methods for Agent
///
/// # Example
/// ```ignore
/// #[tool(name = "get_weather", description = "Get the current weather for a location")]
/// impl WeatherTool {
///     async fn call(&self, input: WeatherInput) -> Result<WeatherOutput, ToolError> {
///         // Implementation
///     }
/// }
///
/// #[derive(Deserialize, JsonSchema)]
/// struct WeatherInput {
///     #[param(description = "The city name")]
///     location: String,
/// }
/// ```
struct ToolArgs {
    name: LitStr,
    description: LitStr,
}

impl Parse for ToolArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: LitStr = input.parse()?;

            match key.to_string().as_str() {
                "name" => name = Some(value),
                "description" => description = Some(value),
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected 'name' or 'description'",
                    ))
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(ToolArgs {
            name: name.ok_or_else(|| input.error("missing 'name' attribute"))?,
            description: description
                .ok_or_else(|| input.error("missing 'description' attribute"))?,
        })
    }
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolArgs);
    let impl_block = parse_macro_input!(item as ItemImpl);

    // Extract struct name
    let struct_name = match &*impl_block.self_ty {
        Type::Path(type_path) => &type_path.path.segments.last().unwrap().ident,
        _ => panic!("Tool can only be applied to impl blocks for named types"),
    };

    // Find the call method
    let call_method = impl_block
        .items
        .iter()
        .find_map(|item| {
            if let syn::ImplItem::Fn(method) = item {
                if method.sig.ident == "call" {
                    Some(method)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .expect("Tool impl must contain an async fn call method");

    // Use the tool name and description from attributes
    let tool_name = args.name.value();
    let tool_description = args.description.value();

    // Extract Input type from the call method's second parameter (first is &self)
    let input_type = call_method
        .sig
        .inputs
        .iter()
        .nth(1)
        .and_then(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                Some(&*pat_type.ty)
            } else {
                None
            }
        })
        .expect("call method must have an input parameter");

    // Validate that call method returns Result
    match &call_method.sig.output {
        ReturnType::Type(_, ty) => {
            if extract_result_ok_type(ty).is_none() {
                panic!("call method must return Result<T, ToolError>");
            }
        }
        _ => panic!("call method must have a return type"),
    };

    // Generate the implementation
    let expanded = quote! {
        #impl_block

        // Implement Tool trait (now includes execution)
        #[async_trait::async_trait]
        impl unai::Tool for #struct_name {
            fn tool_name(&self) -> &'static str {
                #tool_name
            }

            fn definition(&self) -> unai::ToolDefinition {
                unai::ToolDefinition {
                    name: #tool_name.to_string(),
                    description: #tool_description.to_string(),
                    parameters: schemars::schema_for!(#input_type),
                }
            }

            async fn tool_execute(&self, args: serde_json::Value) -> Result<serde_json::Value, unai::ToolError> {
                let input: #input_type = serde_json::from_value(args)
                    .map_err(|e| unai::ToolError::ParseError(e.to_string()))?;

                let output = self.call(input).await?;

                serde_json::to_value(output)
                    .map_err(|e| unai::ToolError::ExecutionError(e.to_string()))
            }
        }
    };

    TokenStream::from(expanded)
}

/// Extract the Ok type from Result<T, E>
fn extract_result_ok_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident == "Result" {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(GenericArgument::Type(ok_type)) = args.args.first() {
                    return Some(ok_type);
                }
            }
        }
    }
    None
}
