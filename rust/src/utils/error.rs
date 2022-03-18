


#[derive(Debug)]
pub enum MSMError {
    InvalidNumDumensions(usize),
    IOError,
    NanOrInf,
}