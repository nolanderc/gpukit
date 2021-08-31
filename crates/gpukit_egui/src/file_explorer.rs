use std::collections::VecDeque;
use std::io;
use std::sync::Arc;

use egui::mutex::Mutex;
use egui::NumExt;

pub struct FileExplorer {
    id: egui::Id,
    initial_path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub enum Event<'a> {
    OpenFile(&'a std::path::Path),
}

#[derive(Debug, Clone)]
struct IoError(Arc<io::Error>);

impl From<io::Error> for IoError {
    fn from(error: io::Error) -> Self {
        IoError(Arc::new(error))
    }
}

#[derive(Debug)]
struct FileExplorerData {
    entries: Result<Arc<[FileExplorerEntry]>, IoError>,
    history: VecDeque<std::path::PathBuf>,
    history_current: usize,
}

#[derive(Debug)]
struct FileExplorerEntry {
    kind: PathKind,
    name: std::ffi::OsString,
    path: std::path::PathBuf,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PathKind {
    Directory,
    File,
}

impl FileExplorer {
    pub fn new(id_source: impl std::hash::Hash) -> Self {
        FileExplorer {
            id: egui::Id::new(id_source),
            initial_path: ".".into(),
        }
    }

    pub fn initial_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.initial_path = path.into();
        self
    }

    pub fn show(self, ui: &mut egui::Ui, mut event_handler: impl FnMut(Event)) {
        let id = ui.make_persistent_id(self.id);

        let explorer = ui
            .memory()
            .id_data_temp
            .get_or_insert_with(id, || {
                Arc::new(Mutex::new(FileExplorerData::open(&self.initial_path)))
            })
            .clone();
        let mut explorer = explorer.lock();

        egui::TopBottomPanel::top("toolbar")
            .frame(egui::Frame::none())
            .show_inside(ui, |ui| {
                ui.vertical(|ui| {
                    ui.add_space(5.0);
                    ui.horizontal(|ui| {
                        let mut nav_button = |label: &str, enabled: bool| {
                            ui.add_sized(
                                egui::vec2(20.0, 20.0),
                                egui::Button::new(label).enabled(enabled),
                            )
                        };

                        if nav_button("\u{2b05}", explorer.num_history_backwards() > 0).clicked() {
                            explorer.history_backward();
                        }

                        if nav_button("\u{27a1}", explorer.num_history_forwards() > 0).clicked() {
                            explorer.history_forward();
                        }

                        if nav_button("\u{2b06}", explorer.path().parent().is_some()).clicked() {
                            let mut path = explorer.path().to_owned();
                            path.pop();
                            explorer.set_path(path);
                        }

                        ui.label(explorer.path().display().to_string());
                    })
                })
            });

        let entries = match &explorer.entries {
            Ok(entries) => entries.clone(),
            Err(error) => {
                let label = egui::Label::new(format!(
                    "failed to load `{}`: {}",
                    explorer.path().display(),
                    error.0
                ))
                .wrap(true);

                ui.allocate_ui_with_layout(
                    ui.available_size(),
                    egui::Layout::left_to_right().with_cross_align(egui::Align::Center),
                    |ui| ui.add_sized(ui.available_size(), label),
                );

                return;
            }
        };

        egui::ScrollArea::auto_sized().show(ui, |ui| {
            let mut selection_rect = None;
            {
                #[derive(Debug, Copy, Clone)]
                struct DragArea {
                    start: egui::Pos2,
                }

                let drag_id = ui.make_persistent_id("drag_selection");
                let response = ui.interact(ui.max_rect(), drag_id, egui::Sense::drag());
                if let Some(position) = response.interact_pointer_pos() {
                    if response.dragged_by(egui::PointerButton::Primary) {
                        let drag = *ui
                            .memory()
                            .data_temp
                            .get_or_insert_with(|| DragArea { start: position });

                        selection_rect = Some(egui::Rect::from_two_pos(drag.start, position));
                    }
                    if response.drag_released() {
                        ui.memory().data_temp.remove::<DragArea>();
                    }
                }
            }

            let icon_size = 64.0;
            let padding = 10.0;
            let width = ui.available_width();
            let columns = (width / (icon_size + padding)).floor().at_least(1.0) as usize;

            egui::Grid::new("grid").show(ui, |ui| {
                for (i, entry) in entries.iter().enumerate() {
                    if i > 0 && i % columns == 0 {
                        ui.end_row();
                    }

                    if explorer_entry_icon(ui, entry, icon_size, selection_rect).double_clicked() {
                        match entry.kind {
                            PathKind::Directory => explorer.set_path(entry.path.clone()),
                            PathKind::File => event_handler(Event::OpenFile(&entry.path)),
                        }
                    }
                }
            });

            if let Some(rect) = selection_rect {
                let style = &ui.style().visuals.widgets.hovered;
                let mut stroke = style.fg_stroke;
                stroke.color = stroke.color.additive();
                stroke.width = 1.0;
                ui.painter()
                    .rect(rect, 2.0, style.bg_fill.additive(), stroke);
            }
        });
    }
}

impl FileExplorerData {
    pub fn open(path: impl AsRef<std::path::Path>) -> FileExplorerData {
        let path = path.as_ref();
        let path = path.canonicalize().unwrap_or_else(|_| path.to_owned());

        let entries = Self::list_entries(&path);

        let mut history = VecDeque::with_capacity(128);
        history.push_back(path);

        FileExplorerData {
            entries,
            history,
            history_current: 0,
        }
    }

    fn path(&self) -> &std::path::Path {
        &self.history[self.history_current]
    }

    fn set_path(&mut self, path: std::path::PathBuf) {
        self.history.drain(self.history_current + 1..);

        if self.history.capacity() == self.history.len() {
            self.history.pop_front();
        } else {
            self.history_current += 1;
        }
        self.history.push_back(path);
        self.update_entries();
    }

    fn num_history_backwards(&self) -> usize {
        self.history_current
    }

    fn num_history_forwards(&self) -> usize {
        self.history.len() - self.history_current - 1
    }

    fn history_backward(&mut self) -> bool {
        if self.num_history_backwards() > 0 {
            self.history_current -= 1;
            self.update_entries();
            true
        } else {
            false
        }
    }

    fn history_forward(&mut self) -> bool {
        if self.num_history_forwards() > 0 {
            self.history_current += 1;
            self.update_entries();
            true
        } else {
            false
        }
    }

    fn update_entries(&mut self) {
        self.entries = Self::list_entries(self.path());
    }

    fn list_entries(dir_path: &std::path::Path) -> Result<Arc<[FileExplorerEntry]>, IoError> {
        let mut entries = std::fs::read_dir(dir_path)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name();
                let path = entry.path().canonicalize().ok()?;
                let kind = if path.is_dir() {
                    PathKind::Directory
                } else {
                    PathKind::File
                };
                Some(FileExplorerEntry { kind, name, path })
            })
            .collect::<Vec<_>>();

        entries.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.name.cmp(&b.name)));

        Ok(entries.into())
    }
}

fn explorer_entry_icon(
    ui: &mut egui::Ui,
    entry: &FileExplorerEntry,
    icon_size: f32,
    selection_rect: Option<egui::Rect>,
) -> egui::Response {
    let padding = 5.0;
    let inner_width = icon_size - 2.0 * padding;

    ui.vertical_centered(|ui| {
        ui.set_min_width(icon_size);
        ui.set_max_width(icon_size);

        let text = entry.name.to_string_lossy().into_owned();

        let layout = ui
            .fonts()
            .layout_multiline(egui::TextStyle::Body, text, inner_width);

        let (outer_rect, response) = ui.allocate_exact_size(
            egui::vec2(
                icon_size,
                padding + icon_size + padding + layout.size.y + padding,
            ),
            egui::Sense::click(),
        );
        let inner_rect = outer_rect.shrink(padding);

        let style = &ui.visuals().widgets;

        if response.hovered() || matches!(selection_rect, Some(sel) if sel.intersects(outer_rect)) {
            ui.painter().rect_filled(
                outer_rect,
                style.hovered.corner_radius,
                style.hovered.bg_fill,
            );
        }

        let painter = ui.painter_at(inner_rect);

        {
            let mut icon_rect = inner_rect;
            icon_rect.set_height(icon_size);
            let icon_color = style.inactive.fg_stroke.color;

            match entry.kind {
                PathKind::Directory => directory_icon(&painter, icon_rect, icon_color),
                PathKind::File => file_icon(&painter, icon_rect, icon_color),
            }
        }

        {
            let mut row_start = 0;
            let mut chars = layout.text.chars();
            let rows = layout.rows.iter().map(|row| {
                let count = row.char_count_excluding_newline();
                chars.by_ref().take(count).for_each(drop);
                let row_end = layout.text.len() - chars.as_str().len();

                let row_text = &layout.text[row_start..row_end];

                row_start = row_end;
                let new_line_count = row.char_count_including_newline() - count;
                chars.by_ref().take(new_line_count).for_each(drop);

                let size = row.rect().size();
                (row_text, size)
            });

            let text_color = style.inactive.text_color();
            let mut middle = inner_rect.center_bottom() - egui::vec2(0.0, layout.size.y);
            for (text, size) in rows {
                painter.text(
                    middle,
                    egui::Align2::CENTER_TOP,
                    text,
                    egui::TextStyle::Body,
                    text_color,
                );
                middle.y += size.y;
            }
        }

        response
    })
    .inner
}

fn file_icon(painter: &egui::Painter, mut area: egui::Rect, color: egui::Color32) {
    let stroke_width = area.size().min_elem() / 20.0;
    area = area.shrink(stroke_width / 2.0);

    let center = area.center();
    let aspect = 2.0 / 3.0;
    let size = if area.aspect_ratio() < aspect {
        egui::vec2(area.width(), area.width() / aspect)
    } else {
        egui::vec2(area.height() * aspect, area.height())
    };

    let half = 0.5 * size;
    let fold_size = 0.4 * size.x;

    painter.add(egui::Shape::Path {
        fill: egui::Color32::TRANSPARENT,
        stroke: egui::Stroke::new(stroke_width, color),
        closed: false,
        points: vec![
            // body
            egui::pos2(center.x + half.x - fold_size, center.y - half.y),
            egui::pos2(center.x - half.x, center.y - half.y),
            egui::pos2(center.x - half.x, center.y + half.y),
            egui::pos2(center.x + half.x, center.y + half.y),
            egui::pos2(center.x + half.x, center.y - half.y + fold_size),
            // fold
            egui::pos2(center.x + half.x - fold_size, center.y - half.y),
            egui::pos2(center.x + half.x - fold_size, center.y - half.y + fold_size),
            egui::pos2(center.x + half.x, center.y - half.y + fold_size),
        ],
    });
}

fn directory_icon(painter: &egui::Painter, mut area: egui::Rect, color: egui::Color32) {
    let stroke_width = area.size().min_elem() / 20.0;
    area = area.shrink(stroke_width / 2.0);

    let center = area.center();
    let aspect = 3.0 / 2.5;
    let size = if area.aspect_ratio() < aspect {
        egui::vec2(area.width(), area.width() / aspect)
    } else {
        egui::vec2(area.height() * aspect, area.height())
    };

    let half = 0.5 * size;
    let fold_height = 0.1 * size.y;

    painter.add(egui::Shape::Path {
        fill: egui::Color32::TRANSPARENT,
        stroke: egui::Stroke::new(stroke_width, color),
        closed: false,
        points: vec![
            // body
            egui::pos2(center.x - half.x, center.y - half.y),
            egui::pos2(center.x - half.x, center.y + half.y),
            egui::pos2(center.x + half.x, center.y + half.y),
            egui::pos2(center.x + half.x, center.y - half.y + fold_height),
            egui::pos2(
                center.x - half.x + 0.4 * size.x,
                center.y - half.y + fold_height,
            ),
            egui::pos2(center.x - half.x + 0.4 * size.x, center.y - half.y),
            egui::pos2(center.x - half.x, center.y - half.y),
        ],
    });
}
