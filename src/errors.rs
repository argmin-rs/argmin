error_chain!{
    errors {
        InvalidParameter(t: String) {
            description("invalid parameter")
            display("invalid parameter: '{}'", t)
        }
    }
}
