pub trait ArgminParameter<T: Clone> {
    fn modify(&mut self) -> T;
}

impl<T: Clone> ArgminParameter<T> for T {
    fn modify(&mut self) -> T {
        self.clone()
    }
}
