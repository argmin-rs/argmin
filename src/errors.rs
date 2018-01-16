error_chain!{
    foreign_links {
        NdarrayLinalg(::ndarray_linalg::error::LinalgError);
    }
    errors {
        InvalidParameter(t: String) {
            description("invalid parameter")
            display("invalid parameter: '{}'", t)
        }
    }
}
