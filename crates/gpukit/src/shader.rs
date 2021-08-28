use anyhow::Context as _;

use inline_str::InlineStr;

pub struct Shader<Stage: shader_stage::ShaderStage> {
    pub(super) label: InlineStr,
    pub(super) spirv: Vec<u32>,
    pub(super) entry_name: InlineStr,
    _phantom: std::marker::PhantomData<Stage>,
}

pub struct ShaderSet<'a> {
    pub vertex: &'a Shader<shader_stage::Vertex>,
    pub fragment: &'a Shader<shader_stage::Fragment>,
}

pub mod shader_stage {
    pub trait ShaderStage: sealed::ShaderStageSealed {}

    mod sealed {
        pub trait ShaderStageSealed {
            const NAGA_STAGE: naga::ShaderStage;
        }
    }

    macro_rules! shader_stage {
        ($ident:ident) => {
            pub struct $ident;
            impl sealed::ShaderStageSealed for $ident {
                const NAGA_STAGE: naga::ShaderStage = naga::ShaderStage::$ident;
            }
            impl ShaderStage for $ident {}
        };
    }

    shader_stage!(Vertex);
    shader_stage!(Fragment);
    shader_stage!(Compute);
}

impl<Stage: shader_stage::ShaderStage> Shader<Stage> {
    fn from_module(module: naga::Module, label: &str) -> anyhow::Result<Self> {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            naga::valid::Capabilities::default(),
        );

        let info = validator
            .validate(&module)
            .context("failed to validate shader")?;

        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                flags: {
                    let mut flags = naga::back::spv::WriterFlags::empty();
                    if cfg!(debug_assertions) {
                        flags |= naga::back::spv::WriterFlags::DEBUG;
                    }
                    flags
                },
                ..Default::default()
            },
        )?;

        Ok(Shader {
            label: label.into(),
            spirv,
            entry_name: "main".into(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn load_glsl(
        path: impl AsRef<std::path::Path>,
        name: Option<&str>,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file_name = path.display().to_string();

        let source = std::fs::read_to_string(path)
            .with_context(|| anyhow!("failed to open file: {}", file_name))?;

        Self::from_glsl(&source, name.unwrap_or_else(|| file_name.as_str()))
            .with_context(|| anyhow!("failed to parse shader: {}", file_name))
    }

    pub fn from_glsl(source: &str, name: &str) -> anyhow::Result<Self> {
        let mut parser = naga::front::glsl::Parser::default();

        let options = naga::front::glsl::Options {
            stage: Stage::NAGA_STAGE,
            defines: Default::default(),
        };

        match parser.parse(&options, source) {
            Ok(module) => Shader::from_module(module, name),
            Err(errors) => {
                let error_iter = errors.into_iter().map(|error| {
                    (
                        error.kind,
                        SourceSpan {
                            start: error.meta.start,
                            end: error.meta.end,
                        },
                    )
                });

                Err(anyhow!(
                    "{}",
                    concat_shader_errors(error_iter, source, name)
                ))
            }
        }
    }
}

struct SourceSpan {
    start: usize,
    end: usize,
}

fn concat_shader_errors<T>(
    errors: impl Iterator<Item = (T, SourceSpan)>,
    source: &str,
    label: &str,
) -> String
where
    T: std::fmt::Display,
{
    use std::fmt::Write;

    let line_endings = source
        .bytes()
        .enumerate()
        .filter(|(_, byte)| *byte == b'\n')
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    // returns the line number and column of the given byte offset
    let source_location = |index: usize| match line_endings.binary_search(&index) {
        Ok(line) | Err(line) => {
            let line_start = if line == 0 {
                0
            } else {
                line_endings[line - 1] + 1
            };
            (line, index - line_start)
        }
    };

    let mut text = String::new();
    for (i, (error, span)) in errors.enumerate() {
        if i > 0 {
            text.push('\n');
        }

        let (start_line, start_column) = source_location(span.start);
        let (end_line, end_column) = source_location(span.end);

        let _ = writeln!(text, "ERROR: {}", error);

        if start_line == end_line {
            let _ = writeln!(
                text,
                " --> {} line {}:{}-{}",
                label,
                1 + start_line,
                1 + start_column,
                1 + end_column
            );
        } else {
            let _ = writeln!(
                text,
                " --> {} line {}:{}-{}:{}",
                label,
                1 + start_line,
                1 + start_column,
                1 + end_line,
                1 + end_column
            );
        }

        let snippet_start = line_endings
            .get(start_line.wrapping_sub(1))
            .copied()
            .map(|offset| offset + 1)
            .unwrap_or(0);
        let snippet_end = line_endings.get(end_line).copied().unwrap_or(source.len());

        let snippet = source[snippet_start..snippet_end].trim_end();

        for line in snippet.lines() {
            let _ = writeln!(text, "  | {}", line);
        }

        if start_line == end_line {
            text.push_str("    ");
            for column in 0..snippet.len() {
                if start_column <= column && column < end_column {
                    text.push('^');
                } else {
                    text.push(' ');
                }
            }
            text.push('\n');
        }
    }

    text
}
