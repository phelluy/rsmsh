use std::fs::File;
/// rsmsh: a Rust library for managing DG meshes

struct Mesh2D {
    vertices: Vec<f64>,
    edges: Vec<usize>,
}

impl Mesh2D {
    // read the data in a gmsh file or return an error
    fn new(gmshfile: &str) -> Result<Mesh2D, std::io::Error> {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        let gmshdata: String = std::fs::read_to_string(gmshfile)?;
        Ok(Mesh2D { vertices, edges })
    }
}

use nom::{
    branch::alt, bytes::complete::{tag, take_until}, character::complete::{alpha1, char, none_of}, combinator::{map, recognize}, error::ErrorKind::Tag, multi::{many0, many1}, sequence::{delimited, preceded, terminated}, Err, IResult, Parser, error::Error,error::ErrorKind,
};
//use nom;

/// all parser functions
// parse a line of the gmsh string
fn parse_line(input: &str) -> IResult<&str, &str> {
    terminated(take_until("\n"), tag("\n"))(input)
}

fn parse_line_usizes(input: &str) -> IResult<&str, Vec<usize>> {
    let res = parse_line(input);
    match res {
        Ok((rest, line)) => {
            let v: Result<Vec<usize>, _> = line
                .split_whitespace()
                .map(|x| x.parse::<usize>())
                .collect();
            match v {
                Ok(v) => Ok((rest, v)),
                Err(_) => Err(nom::Err::Error(Error::new(input, ErrorKind::Tag))),
            }
        }
        Err(e) => Err(e),
    }
}

fn parse_line_f64s(input: &str) -> IResult<&str, Vec<f64>> {
    map(parse_line, |line| {
        line.split_whitespace()
            .map(|x| x.parse::<f64>().unwrap())
            .collect()
    })(input)
}

fn parse_nodes_header(input: &str) -> IResult<&str, (usize, usize, usize, usize)> {
    map(parse_line_usizes, |v| {
        let numblocks = v[0];
        let numnodes = v[1];
        let minnode = v[2];
        let maxnode = v[3];
        (v[0], v[1], v[2], v[3])
    })(input)
}

// fn parse_nodes_block(input: &str) -> IResult<&str, Vec<f64>> {
//     let res = parse_line_usizes(input);

// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mesh2d() {
        let mesh = Mesh2D::new("geo/square.msh").unwrap();
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.edges.len(), 0);
    }

    #[test]
    fn test_parse_line() {
        let line = "1 2 3 4 5 6 7 8 9 10\n a b c ";
        let (rest, parsed) = parse_line(line).unwrap();
        assert_eq!(rest, " a b c ");
        assert_eq!(parsed, "1 2 3 4 5 6 7 8 9 10");
    }

    #[test]
    fn test_parse_line_usizes() {
        let line = "1 2 3 4 5 6 7 8 9 10\n";
        println!("{:?}", parse_line_usizes(line));
        let (rest, parsed) = parse_line_usizes(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(parsed, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_parse_line_f64s() {
        let line = "1 2.0 3.0 4.0 5.0 6  7.0 8.0 9.0 10.0\n";
        let (rest, parsed) = parse_line_f64s(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(
            parsed,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_parse_nodes_header() {
        let line = "10 0 3 0\n";
        let (rest, parsed) = parse_nodes_header(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(parsed, (10, 0, 3, 0));
    }
}
