use std::rc::Rc;

pub struct InlineStr(InlineStrRepr);

pub const INLINE_LEN: usize = 3 * std::mem::size_of::<usize>() - 2;

#[derive(Clone)]
enum InlineStrRepr {
    Inline { len: u8, data: [u8; INLINE_LEN] },
    Heap(Rc<str>),
}

impl InlineStr {
    pub fn new(text: &str) -> Self {
        if text.len() <= INLINE_LEN {
            let mut data = [0u8; INLINE_LEN];
            data[..text.len()].copy_from_slice(text.as_bytes());
            InlineStr(InlineStrRepr::Inline {
                len: text.len() as u8,
                data,
            })
        } else {
            InlineStr(InlineStrRepr::Heap(text.into()))
        }
    }
}

impl From<&str> for InlineStr {
    fn from(text: &str) -> Self {
        InlineStr::new(text)
    }
}

impl From<Rc<str>> for InlineStr {
    fn from(text: Rc<str>) -> Self {
        InlineStr(InlineStrRepr::Heap(text))
    }
}

impl std::ops::Deref for InlineStr {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match &self.0 {
            InlineStrRepr::Inline { len, data } => unsafe {
                // SAFETY: the only way of initializing `data` is using a valid UTF-8 `str`.
                std::str::from_utf8_unchecked(&data[..*len as usize])
            },
            InlineStrRepr::Heap(text) => text,
        }
    }
}

impl std::borrow::Borrow<str> for InlineStr {
    fn borrow(&self) -> &str {
        self
    }
}

impl std::hash::Hash for InlineStr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::ops::Deref;
        self.deref().hash(state);
    }
}

impl PartialEq for InlineStr {
    fn eq(&self, other: &Self) -> bool {
        use std::ops::Deref;
        self.deref() == other.deref()
    }
}

impl PartialEq<str> for InlineStr {
    fn eq(&self, other: &str) -> bool {
        use std::ops::Deref;
        self.deref() == other
    }
}

impl PartialEq<InlineStr> for &str {
    fn eq(&self, other: &InlineStr) -> bool {
        use std::ops::Deref;
        *self == other.deref()
    }
}

impl Eq for InlineStr {}

impl PartialOrd for InlineStr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::ops::Deref;
        self.deref().partial_cmp(other.deref())
    }
}

impl PartialOrd<str> for InlineStr {
    fn partial_cmp(&self, other: &str) -> Option<std::cmp::Ordering> {
        use std::ops::Deref;
        self.deref().partial_cmp(other)
    }
}

impl PartialOrd<InlineStr> for &str {
    fn partial_cmp(&self, other: &InlineStr) -> Option<std::cmp::Ordering> {
        use std::ops::Deref;
        (*self).partial_cmp(other.deref())
    }
}

impl Ord for InlineStr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::ops::Deref;
        self.deref().cmp(other.deref())
    }
}

impl std::fmt::Debug for InlineStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::ops::Deref;
        self.deref().fmt(f)
    }
}

impl std::fmt::Display for InlineStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::ops::Deref;
        self.deref().fmt(f)
    }
}
