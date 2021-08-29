use anyhow::Context as _;

pub struct Shader {
    module: wgpu::ShaderModule,
}

pub struct ShaderEntry<'a> {
    pub(crate) module: &'a wgpu::ShaderModule,
    pub(crate) entry_point: &'a str,
}

#[derive(Debug, Copy, Clone)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

pub struct ShaderDescriptor<'a> {
    pub label: &'a str,
    pub source: ShaderSource<'a>,
}

pub enum ShaderSource<'a> {
    Glsl(&'a str, ShaderStage),
    Wgsl(&'a str),
}

pub struct ShaderBuilder<'a> {
    context: &'a crate::Context,
    label: &'a str,
}

impl Shader {
    pub fn new(context: &crate::Context, desc: &ShaderDescriptor) -> anyhow::Result<Shader> {
        let result = match desc.source {
            ShaderSource::Glsl(source, stage) => Self::parse_glsl(source, stage, desc.label),
            ShaderSource::Wgsl(source) => Self::parse_wgsl(source),
        };

        let module = result.with_context(|| anyhow!("failed to parse shader `{}`", desc.label))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            naga::valid::Capabilities::default(),
        );

        let info = validator
            .validate(&module)
            .with_context(|| anyhow!("failed to validate shader `{}`", desc.label))?;

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

        let module = context
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some(desc.label),
                source: wgpu::ShaderSource::SpirV(spirv.into()),
            });

        Ok(Shader { module })
    }

    fn parse_glsl(source: &str, stage: ShaderStage, label: &str) -> anyhow::Result<naga::Module> {
        let mut parser = naga::front::glsl::Parser::default();

        let options = naga::front::glsl::Options {
            stage: stage.into(),
            defines: Default::default(),
        };

        match parser.parse(&options, source) {
            Ok(module) => Ok(module),
            Err(errors) => {
                let file = codespan_reporting::files::SimpleFile::new(label, source);
                let mut writer = codespan_reporting::term::termcolor::Buffer::ansi();
                let config = codespan_reporting::term::Config::default();

                for error in errors {
                    use codespan_reporting::diagnostic::{Diagnostic, Label};

                    let start = error.meta.start;
                    let end = error.meta.end;
                    let diagnostic = Diagnostic::error()
                        .with_message(error.kind.to_string())
                        .with_labels(vec![Label::primary((), start..end)]);
                    codespan_reporting::term::emit(&mut writer, &config, &file, &diagnostic)
                        .unwrap();
                }

                let text = String::from_utf8(writer.into_inner()).unwrap();
                Err(anyhow!("{}", text))
            }
        }
    }

    fn parse_wgsl(source: &str) -> anyhow::Result<naga::Module> {
        naga::front::wgsl::parse_str(source).map_err(|err| {
            let message = err.emit_to_string(source);
            anyhow!("{}", message)
        })
    }

    pub fn entry<'a>(&'a self, name: &'a str) -> ShaderEntry<'a> {
        ShaderEntry {
            module: &self.module,
            entry_point: name,
        }
    }
}

impl<'a> ShaderBuilder<'a> {
    pub(crate) fn new(context: &'a crate::Context, label: &'a str) -> Self {
        ShaderBuilder { context, label }
    }

    fn from_source(self, source: ShaderSource) -> anyhow::Result<Shader> {
        Shader::new(
            self.context,
            &ShaderDescriptor {
                label: self.label,
                source,
            },
        )
    }

    pub fn from_glsl(self, source: &str, stage: ShaderStage) -> anyhow::Result<Shader> {
        self.from_source(ShaderSource::Glsl(source, stage))
    }

    pub fn from_wgsl(self, source: &str) -> anyhow::Result<Shader> {
        self.from_source(ShaderSource::Wgsl(source))
    }

    pub fn load(self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Shader> {
        let path = path.as_ref();

        let load_context = || format!("failed to load shader `{}`", path.display());

        let kind = ShaderKind::infer_from_path(path)
            .with_context(|| format!("failed to infer shader kind"))
            .with_context(load_context)?;

        let text = std::fs::read_to_string(path).with_context(load_context)?;

        self.from_source(kind.source(&text))
    }
}

enum ShaderKind {
    Glsl(ShaderStage),
    Wgsl,
}

impl ShaderKind {
    fn infer_from_path(path: &std::path::Path) -> anyhow::Result<ShaderKind> {
        let extension = path.extension().and_then(|ext| ext.to_str());

        match extension {
            Some("vert") => Ok(ShaderKind::Glsl(ShaderStage::Vertex)),
            Some("frag") => Ok(ShaderKind::Glsl(ShaderStage::Fragment)),
            Some("comp") => Ok(ShaderKind::Glsl(ShaderStage::Compute)),
            Some("wgsl") => Ok(ShaderKind::Wgsl),
            Some(extension) => {
                return Err(anyhow!("unknown shader path extension: `{}`", extension))
            }
            None => return Err(anyhow!("shader path missing extension")),
        }
    }

    pub fn source(self, source: &str) -> ShaderSource {
        match self {
            ShaderKind::Glsl(stage) => ShaderSource::Glsl(source, stage),
            ShaderKind::Wgsl => ShaderSource::Wgsl(source),
        }
    }
}

impl From<ShaderStage> for naga::ShaderStage {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ShaderStage::Compute => naga::ShaderStage::Compute,
        }
    }
}

impl From<naga::ShaderStage> for ShaderStage {
    fn from(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
        }
    }
}
