pub trait ArgminParameter {
    fn modify(&mut self) -> &mut Self;
}

impl ArgminParameter for Vec<f64> {
    fn modify(&mut self) -> &mut Self {
        self
    }
}
