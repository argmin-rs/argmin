pub trait ArgminParameter<T: Clone> {
    fn modify(&mut self, &Option<T>, &Option<T>, &Option<&Fn(&T) -> bool>) -> T;
}

impl<T: Clone> ArgminParameter<T> for T {
    fn modify(
        &mut self,
        lower_bound: &Option<T>,
        upper_bound: &Option<T>,
        constraint: &Option<&Fn(&T) -> bool>,
    ) -> T {
        self.clone()
    }
}
