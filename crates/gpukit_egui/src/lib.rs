mod renderer;

pub use egui;
pub use renderer::{Renderer, RendererDescriptor};

use gpukit::{wgpu, winit};
use std::sync::Arc;

pub struct Gui {
    window: Arc<winit::window::Window>,
    window_state: WindowState,

    ctx: egui::CtxRef,
    input: egui::RawInput,
    output: egui::Output,

    renderer: Renderer,
    meshes: Vec<egui::ClippedMesh>,

    start_time: std::time::Instant,
}

pub struct GuiDescriptor {
    pub context: Arc<gpukit::Context>,
    pub window: Arc<winit::window::Window>,
    pub target_format: wgpu::TextureFormat,
}

struct WindowState {
    last_cursor_position: egui::Pos2,
    modifiers: egui::Modifiers,
}

impl Gui {
    pub fn new(desc: GuiDescriptor) -> anyhow::Result<Gui> {
        let GuiDescriptor {
            context,
            window,
            target_format,
        } = desc;

        let pixels_per_point = window.scale_factor() as f32;
        let size = window.inner_size();

        let renderer = Renderer::new(RendererDescriptor {
            context,
            target_format,
            size: [size.width, size.height],
            pixels_per_point,
        })?;

        let start_time = std::time::Instant::now();

        Ok(Gui {
            window,
            window_state: WindowState {
                last_cursor_position: egui::pos2(0.0, 0.0),
                modifiers: egui::Modifiers::default(),
            },

            ctx: egui::CtxRef::default(),
            renderer,
            input: egui::RawInput {
                screen_rect: Some(Self::egui_rect_from_size(
                    size.to_logical(pixels_per_point as f64),
                )),
                pixels_per_point: Some(pixels_per_point),
                ..Default::default()
            },
            output: egui::Output::default(),
            meshes: Vec::new(),

            start_time,
        })
    }

    fn update_size(&mut self, size: winit::dpi::PhysicalSize<u32>, scale_factor: f64) {
        let pixels_per_point = scale_factor as f32;
        self.renderer.set_size(size.into(), pixels_per_point);

        let rect = Self::egui_rect_from_size(size.to_logical(scale_factor));
        self.input.screen_rect = Some(rect);
        self.input.pixels_per_point = Some(pixels_per_point);
    }

    pub fn handle_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::WindowEvent;
        match event {
            WindowEvent::Resized(size) => {
                self.update_size(*size, self.window.scale_factor());
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                new_inner_size,
            } => {
                self.update_size(**new_inner_size, *scale_factor);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let logical_position = position.to_logical(self.window.scale_factor());
                let egui_position = Self::egui_pos_from_winit(logical_position);
                self.window_state.last_cursor_position = egui_position;
                self.input
                    .events
                    .push(egui::Event::PointerMoved(egui_position));
            }
            WindowEvent::MouseInput { button, state, .. } => {
                if let Some(button) = Self::egui_button_from_winit(*button) {
                    self.input.events.push(egui::Event::PointerButton {
                        pos: self.window_state.last_cursor_position,
                        button,
                        pressed: *state == winit::event::ElementState::Pressed,
                        modifiers: self.window_state.modifiers,
                    })
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.window_state.modifiers = Self::egui_modifiers_from_winit(*modifiers);
            }
            WindowEvent::CursorLeft { .. } => self.input.events.push(egui::Event::PointerGone),

            WindowEvent::ReceivedCharacter(ch) => {
                if !ch.is_control() {
                    self.input.events.push(egui::Event::Text(ch.to_string()));
                }
            }
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(key) = input.virtual_keycode.and_then(Self::egui_key_from_winit) {
                    self.input.events.push(egui::Event::Key {
                        key,
                        pressed: input.state == winit::event::ElementState::Pressed,
                        modifiers: self.window_state.modifiers,
                    });
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, _) => {}
                winit::event::MouseScrollDelta::PixelDelta(delta) => {
                    let scale = self.window.scale_factor() as f32;
                    self.input.scroll_delta =
                        egui::vec2(delta.x as f32 / scale, delta.y as f32 / scale);
                }
            },

            _ => {}
        }
    }

    pub fn update<T, E>(
        &mut self,
        state: &mut T,
        draw: impl FnOnce(egui::CtxRef, &mut T) -> Result<(), E>,
    ) -> Result<(), E> {
        self.input.time = Some(self.start_time.elapsed().as_secs_f64());

        self.ctx.begin_frame(self.input.take());
        self.ctx.set_visuals(egui::Visuals {
            window_shadow: egui::paint::Shadow {
                extrusion: 10.0,
                color: egui::Color32::from_rgba_premultiplied(0, 0, 0, 100),
            },
            ..Default::default()
        });

        draw(self.ctx.clone(), state)?;

        let (output, shapes) = self.ctx.end_frame();
        self.meshes = self.ctx.tessellate(shapes);

        if self.output.cursor_icon != output.cursor_icon {
            self.update_cursor_icon(output.cursor_icon);
        }

        self.output = output;

        Ok(())
    }

    fn update_cursor_icon(&mut self, new_icon: egui::CursorIcon) {
        if let Some(icon) = Self::winit_icon_from_egui(new_icon) {
            self.window.set_cursor_visible(true);
            self.window.set_cursor_icon(icon);
        } else {
            self.window.set_cursor_visible(false);
        }
    }

    pub fn needs_redraw(&self) -> bool {
        self.output.needs_repaint
    }

    pub fn render<'encoder>(&'encoder mut self, rpass: &mut wgpu::RenderPass<'encoder>) {
        self.renderer
            .render(rpass, &self.meshes, &self.ctx.texture())
    }

    fn egui_rect_from_size(size: winit::dpi::LogicalSize<u32>) -> egui::Rect {
        let position = egui::pos2(0.0, 0.0);
        let size = Self::egui_size_from_winit(size);
        egui::Rect::from_min_size(position, size)
    }

    fn egui_size_from_winit(position: winit::dpi::LogicalSize<u32>) -> egui::Vec2 {
        egui::vec2(position.width as f32, position.height as f32)
    }

    fn egui_pos_from_winit(position: winit::dpi::LogicalPosition<f64>) -> egui::Pos2 {
        egui::pos2(position.x as f32, position.y as f32)
    }

    fn egui_button_from_winit(button: winit::event::MouseButton) -> Option<egui::PointerButton> {
        match button {
            winit::event::MouseButton::Left => Some(egui::PointerButton::Primary),
            winit::event::MouseButton::Right => Some(egui::PointerButton::Secondary),
            winit::event::MouseButton::Middle => Some(egui::PointerButton::Middle),
            winit::event::MouseButton::Other(_) => None,
        }
    }

    fn egui_modifiers_from_winit(modifiers: winit::event::ModifiersState) -> egui::Modifiers {
        use winit::event::ModifiersState as Modifier;
        egui::Modifiers {
            alt: modifiers.contains(Modifier::ALT),
            ctrl: modifiers.contains(Modifier::CTRL),
            shift: modifiers.contains(Modifier::SHIFT),
            mac_cmd: cfg!(target_os = "macos") && modifiers.contains(Modifier::LOGO),
            command: if cfg!(target_os = "macos") {
                modifiers.contains(Modifier::LOGO)
            } else {
                modifiers.contains(Modifier::CTRL)
            },
        }
    }

    fn egui_key_from_winit(key: winit::event::VirtualKeyCode) -> Option<egui::Key> {
        use winit::event::VirtualKeyCode as KeyCode;

        let egui_key = match key {
            KeyCode::Down => egui::Key::ArrowDown,
            KeyCode::Left => egui::Key::ArrowLeft,
            KeyCode::Right => egui::Key::ArrowRight,
            KeyCode::Up => egui::Key::ArrowUp,
            KeyCode::Escape => egui::Key::Escape,
            KeyCode::Tab => egui::Key::Tab,
            KeyCode::Back => egui::Key::Backspace,
            KeyCode::Return => egui::Key::Enter,
            KeyCode::Space => egui::Key::Space,
            KeyCode::Insert => egui::Key::Insert,
            KeyCode::Delete => egui::Key::Delete,
            KeyCode::Home => egui::Key::Home,
            KeyCode::End => egui::Key::End,

            KeyCode::Key1 | KeyCode::Numpad1 => egui::Key::Num1,
            KeyCode::Key2 | KeyCode::Numpad2 => egui::Key::Num2,
            KeyCode::Key3 | KeyCode::Numpad3 => egui::Key::Num3,
            KeyCode::Key4 | KeyCode::Numpad4 => egui::Key::Num4,
            KeyCode::Key5 | KeyCode::Numpad5 => egui::Key::Num5,
            KeyCode::Key6 | KeyCode::Numpad6 => egui::Key::Num6,
            KeyCode::Key7 | KeyCode::Numpad7 => egui::Key::Num7,
            KeyCode::Key8 | KeyCode::Numpad8 => egui::Key::Num8,
            KeyCode::Key9 | KeyCode::Numpad9 => egui::Key::Num9,
            KeyCode::Key0 | KeyCode::Numpad0 => egui::Key::Num0,

            KeyCode::A => egui::Key::A,
            KeyCode::B => egui::Key::B,
            KeyCode::C => egui::Key::C,
            KeyCode::D => egui::Key::D,
            KeyCode::E => egui::Key::E,
            KeyCode::F => egui::Key::F,
            KeyCode::G => egui::Key::G,
            KeyCode::H => egui::Key::H,
            KeyCode::I => egui::Key::I,
            KeyCode::J => egui::Key::J,
            KeyCode::K => egui::Key::K,
            KeyCode::L => egui::Key::L,
            KeyCode::M => egui::Key::M,
            KeyCode::N => egui::Key::N,
            KeyCode::O => egui::Key::O,
            KeyCode::P => egui::Key::P,
            KeyCode::Q => egui::Key::Q,
            KeyCode::R => egui::Key::R,
            KeyCode::S => egui::Key::S,
            KeyCode::T => egui::Key::T,
            KeyCode::U => egui::Key::U,
            KeyCode::V => egui::Key::V,
            KeyCode::W => egui::Key::W,
            KeyCode::X => egui::Key::X,
            KeyCode::Y => egui::Key::Y,
            KeyCode::Z => egui::Key::Z,

            _ => return None,
        };

        Some(egui_key)
    }

    fn winit_icon_from_egui(icon: egui::CursorIcon) -> Option<winit::window::CursorIcon> {
        use winit::window::CursorIcon as Icon;

        let winit_icon = match icon {
            egui::CursorIcon::None => return None,

            egui::CursorIcon::Default => Icon::Default,
            egui::CursorIcon::ContextMenu => Icon::ContextMenu,
            egui::CursorIcon::Help => Icon::Help,
            egui::CursorIcon::PointingHand => Icon::Hand,
            egui::CursorIcon::Progress => Icon::Progress,
            egui::CursorIcon::Wait => Icon::Wait,
            egui::CursorIcon::Cell => Icon::Cell,
            egui::CursorIcon::Crosshair => Icon::Crosshair,
            egui::CursorIcon::Text => Icon::Text,
            egui::CursorIcon::VerticalText => Icon::VerticalText,
            egui::CursorIcon::Alias => Icon::Alias,
            egui::CursorIcon::Copy => Icon::Copy,
            egui::CursorIcon::Move => Icon::Move,
            egui::CursorIcon::NoDrop => Icon::NoDrop,
            egui::CursorIcon::NotAllowed => Icon::NotAllowed,
            egui::CursorIcon::Grab => Icon::Grab,
            egui::CursorIcon::Grabbing => Icon::Grabbing,
            egui::CursorIcon::AllScroll => Icon::AllScroll,
            egui::CursorIcon::ResizeHorizontal => Icon::EwResize,
            egui::CursorIcon::ResizeNeSw => Icon::NeswResize,
            egui::CursorIcon::ResizeNwSe => Icon::NwseResize,
            egui::CursorIcon::ResizeVertical => Icon::NsResize,
            egui::CursorIcon::ZoomIn => Icon::ZoomIn,
            egui::CursorIcon::ZoomOut => Icon::ZoomOut,
        };

        Some(winit_icon)
    }
}
