use std::{
    fs::File,
    collections::HashMap,
    io::{Read},
    error::Error,
    self};
use regex::Regex;

mod errors;
pub use errors::*;

pub type NNConfig = Vec<HashMap<String, String>>; // Currently support only for sequential NNs

pub fn parse_file(filename: &str) -> Result<NNConfig, Box<dyn Error>> {
    let mut file = File::open(filename)?;
    let mut lines = String::new();
    file.read_to_string(&mut lines)?;

    let lines: Vec<String> = lines.split_terminator('\n').into_iter().filter(|l| {
        let pretty_string = l.trim();
        !pretty_string.starts_with('#') && !pretty_string.is_empty()
    }).map(|l| {
        l.split_whitespace().collect::<String>()
    }).collect();

    let section_start_re = Regex::new(r"^\[[A-Za-z]*\]$").unwrap();
    let statement_re = Regex::new(r"^[A-Za-z|_]*=\w*").unwrap();
    let mut config: NNConfig = vec![];

    for line in lines {
        let line = line.as_ref();
        if section_start_re.is_match(line) { // new object started, creating template
            let mut object = HashMap::new();
            let substr = &line[1..line.len()-1];
            object.insert(String::from("type"), String::from(substr));

            config.push(object);
        } else if statement_re.is_match(line){
            let (key, value) = match line.split_once("=") {
                Some(s) => s,
                None => return ParseError::generate(line)
            };
            if let Some(object) = config.last_mut() {
                object.insert(String::from(key), String::from(value));
            } else {
                return ParseError::generate("Internal parse error");
            }
        } else {
            return ParseError::generate(line);
        }
    }
    Ok(config)
}

pub fn parse_numerical_field<T>(config: &HashMap<String, String>, key: &str, mandatory: bool, default: Option<T>) -> Result<Option<T>, DeserializationError> 
where T: std::str::FromStr
{
    let str_value = config.get(key);
    match str_value {
        Some(x) => match x.parse::<T>() {
            Ok(value) => return Ok(Some(value)),
            Err(_) => return Err(DeserializationError{description: format!("Couldnt parse '{}' for key '{}'", x, key)})
        },
        None => match mandatory {
            true => return Err(DeserializationError{description: format!("Key '{}' is mandatory", key)}),
            false => return Ok(default)
        }
    }
}

pub fn parse_list_field<T>(config: &HashMap<String, String>, key: &str, layer_name: &str) -> Result<Vec<T>, DeserializationError>
where T: std::str::FromStr
{
    config.get(key)
        .ok_or(DeserializationError{description: format!("Field '{}' is mandatory for {}", key, layer_name)})?
        .split(',')
        .map(|x| {
            x.trim().parse::<T>()
            .map_err(|_e| {
                DeserializationError{description: format!("Couldnt parse value '{}' in field '{}' of {}", x, key, layer_name)}
            })
        })
        .collect()
}

pub fn ensure_positive(value: usize, key: &str, layer: &str) -> Result<(), DeserializationError> {
    match value < 1 {
        true => return Err(DeserializationError{description: format!("Field '{}' of {} should be positive", key, layer)}),
        _ => Ok(())
    }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn file_not_exists() {
        let filename = String::from("./cfgs/tests/base_fake.cfg");
        let result = parse_file(&filename);
        match result {
            Ok(_) => assert!(false),
            Err(err_ref) => match err_ref.downcast::<io::Error>() {
                Err(_) => assert!(false),
                Ok(err) => match err.kind() {
                    io::ErrorKind::NotFound => assert!(true),
                    _ => assert!(false)
                }
            }
        }
    }

    #[test]
    fn success_parsed_base() {
        use md5;
        use itertools::Itertools; 
        let filename = String::from("../cfgs/tests/base.cfg");
        let result = parse_file(&filename);
        match result {
            Ok(config) => {
                let mut concated = String::new();
                for obj in config {
                    for (k, v) in obj.iter().sorted() {
                        concated.push_str(
                            format!("{}={}", k, v).as_ref()
                        );
                    }
                }

                println!("{:x}|{}", md5::compute(concated.as_bytes()), concated);
                assert_eq!(format!("{:x}", md5::compute(concated)), "d56b854d6056da56e9e2d464857b324c");
            },
            Err(_) => assert!(false)
        }
    }
}