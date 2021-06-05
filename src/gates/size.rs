use enum_dispatch::enum_dispatch;

/// Trait to indicate the size (number of qudits) of a gate.
#[enum_dispatch]
pub trait Size {
    /// Get the size of a gate in number of qudits
    fn get_size(&self) -> usize;
}
