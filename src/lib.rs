//use std::fs::File;
/// rsmsh: a Rust library for managing DG meshes

#[derive(Debug, Clone)]
pub enum BoundaryType {
    Dirichlet, // imposed data
    #[allow(dead_code)]
    Neumann, // wall
    Elem(usize), // internal boundary: elem number
    LocEdge(usize), // local edge number
}

#[derive(Debug, Clone)]
pub struct Mesh2D {
    nbnodes: usize,                        // number of nodes
    nbelems: usize,                        // number of elements
    nbedges: usize,                        // number of edges
    vertices: Vec<(f64, f64, f64)>,        // list of vertices (x, y)
    elems: Vec<Vec<usize>>,                // list of elements (list of nodes)
    edges: Vec<[usize; 3]>,                // list of edges (list of nodes)
    elem2elem: Vec<Vec<BoundaryType>>,     // elem->elem connectivity
    edge2elem: Vec<(usize, BoundaryType)>, // edge->elem connectivity
    edge2edge: Vec<(usize, BoundaryType)>, // edge-> local edge connectivity
    length: Vec<f64>,                      // length of edges
    surface: Vec<f64>,                     // surface of elements
    normal: Vec<(f64, f64)>,               // unit normal to edges
    bounding_box: (f64, f64, f64, f64),    // bounding box
    min_length: f64,                       // minimum edge length
}

use core::panic;
use std::io::Write;
use std::vec;

impl Mesh2D {
    // read the data in a gmsh file or return an error
    pub fn new(gmshfile: &str) -> Mesh2D {
        let gmshdata: String = std::fs::read_to_string(gmshfile).unwrap();
        let (_, (vertices, elems)) = parse_nodes_elems(&gmshdata).unwrap();
        let (edges, elem2elem, edge2elem, edge2edge) = build_connectivity(&elems);
        let mut length = vec![];
        let mut surface = vec![];
        let mut normal = vec![];
        for elem in &elems {
            let (x1, y1, _) = vertices[elem[0]];
            let (x2, y2, _) = vertices[elem[1]];
            let (x3, y3, _) = vertices[elem[2]];
            let s = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
            assert!(s > 0.0);
            surface.push(s.abs());
        }
        let mut min_length = std::f64::MAX;
        for edge in &edges {
            let (x1, y1, _) = vertices[edge[0]];
            let (x2, y2, _) = vertices[edge[1]];
            let ll = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
            length.push(ll);
            if ll < min_length {
                min_length = ll;
            }
            let nx = y2 - y1;
            let ny = x1 - x2;
            let n = (nx * nx + ny * ny).sqrt();
            normal.push((nx / n, ny / n));
        }
        let mut xmin = std::f64::MAX;
        let mut xmax = std::f64::MIN;
        let mut ymin = std::f64::MAX;
        let mut ymax = std::f64::MIN;
        for (x, y, _) in &vertices {
            if *x < xmin {
                xmin = *x;
            }
            if *x > xmax {
                xmax = *x;
            }
            if *y < ymin {
                ymin = *y;
            }
            if *y > ymax {
                ymax = *y;
            }
        }

        Mesh2D {
            nbnodes: vertices.len(),
            nbelems: elems.len(),
            nbedges: edges.len(),
            vertices,
            elems,
            edges,
            elem2elem,
            edge2elem,
            edge2edge,
            length,
            surface,
            normal,
            bounding_box: (xmin, xmax, ymin, ymax),
            min_length,
        }
    }

    pub fn get_nbelems(&self) -> usize {
        self.nbelems
    }

    pub fn get_center(&self, i: usize) -> (f64, f64) {
        let elem = &self.elems[i];
        let mut x = 0.0;
        let mut y = 0.0;
        let mut w = 0.0;
        for node in elem {
            let (x1, y1, _) = self.vertices[*node];
            x += x1;
            y += y1;
            w += 1.0;
        }
        (x / w, y / w)
    }

    pub fn get_surf(&self, i: usize) -> f64 {
        self.surface[i]
    }

    pub fn get_vois(&self, i: usize, loc_edge: usize) -> BoundaryType {
        self.elem2elem[i][loc_edge].clone()
    }

    // get length of edge loc_edge of element i
    pub fn get_length(&self, i: usize, loc_edge: usize) -> f64 {
        let (i1, i2) = (self.elems[i][loc_edge], self.elems[i][(loc_edge + 1) % 3]);
        let (x1, x2) = (self.vertices[i1], self.vertices[i2]);
        ((x2.0 - x1.0).powi(2) + (x2.1 - x1.1).powi(2)).sqrt()
    }

    // get outward unit normal to edge loc_edge of element i
    pub fn get_normal(&self, i: usize, loc_edge: usize) -> [f64; 2] {
        let (i1, i2) = (self.elems[i][loc_edge], self.elems[i][(loc_edge + 1) % 3]);
        let (x1, x2) = (self.vertices[i1], self.vertices[i2]);
        let nx = x2.1 - x1.1;
        let ny = x1.0 - x2.0;
        let n = (nx * nx + ny * ny).sqrt();
        [nx / n, ny / n]
    }

    pub fn get_edge_center(&self, iel: usize, iloc: usize) -> (f64, f64) {
        let (i1,i2) = (self.elems[iel][iloc], self.elems[iel][(iloc+1)%3]);
        let (x1, y1, _) = self.vertices[i1];
        let (x2, y2, _) = self.vertices[i2];
        ((x1+x2)/2., (y1+y2)/2.)
    }

    // the perimeter of elem i
    pub fn get_perimeter(&self, i: usize) -> f64 {
        let mut p = 0.0;
        for j in 0..3 {
            p += self.get_length(i, j);
        }
        p
    }

    // this function may fail
    // the error contains a static str
    pub fn make_periodic(&mut self) -> Result<(), &'static str> {
        let (xmin, _, ymin, _) = self.bounding_box;
        let h = self.min_length / 3.;
        // create a hashmap of the boundary edges
        // key = ((intx,inty),(nx,ny)) value = num of the edge
        // (intx,inty) are the coordinates of the middle of the edge converted to an integer
        // on a grid of step h
        // (nx,ny) = (1,0), (0,1), (-1,0), (0,-1) are the normal to the edges converted also to integers
        let mut dx = std::i32::MIN;
        let mut dy = std::i32::MIN;
        let mut border_edges = std::collections::HashMap::new();
        for ((i, edge), edge2elem) in self.edges.iter().enumerate().zip(self.edge2elem.iter()) {
            // the edge is a boundary edge
            if let (_, Dirichlet | Neumann) = edge2elem {
                let (x1, y1, _) = self.vertices[edge[0]];
                let (x2, y2, _) = self.vertices[edge[1]];
                let (vx, vy) = self.normal[i];
                let (vx, vy) = (vx.round() as i32, vy.round() as i32);
                let intx = (((x1 + x2) - xmin) / 2. / h).round() as i32;
                let inty = (((y1 + y2) / 2. - ymin) / h).round() as i32;
                if dx < intx {
                    dx = intx;
                }
                if dy < inty {
                    dy = inty;
                }
                border_edges.insert(((intx, inty), (vx, vy)), i);
            }
        }
        //println!("{:?}", border_edges);
        println!("Making the mesh periodic");
        println!("nbelems={}", self.nbelems);
        println!("dx = {}, dy = {}", dx, dy);
        // create a list of matching periodic edges
        // the first edge in the pair corresponds to a negative normal
        let mut paired_edges: Vec<(usize, usize)> = vec![];
        for ((intx, inty), (nx, ny)) in border_edges.keys() {
            if *nx == -1 {
                // find the matching edge with nx = 1
                let edge = border_edges
                    .get(&((intx + dx, *inty), (1, 0)))
                    .ok_or("periodic right edge not found")?;
                paired_edges.push((border_edges[&((*intx, *inty), (-1, 0))], *edge));
            }
            if *ny == -1 {
                // find the matching edge with ny = 1
                let edge = border_edges
                    .get(&((*intx, inty + dy), (0, 1)))
                    .ok_or("periodic top edge not found")?;

                paired_edges.push((border_edges[&((*intx, *inty), (0, -1))], *edge));
            }
        }
        //println!("{:?}", paired_edges);
        let new_nbedges = self.nbedges - paired_edges.len();
        // update conserved edges
        for (i, j) in paired_edges {
            self.edge2elem[j] = (self.edge2elem[j].0, Elem(self.edge2elem[i].0));
            self.edge2edge[j] = (self.edge2edge[j].0, LocEdge(self.edge2edge[i].0));
            //println!("update edge {} with edge {}", j, i);
        }
        let mut new_edges = vec![];
        let mut new_edge2elem = vec![];
        let mut new_edge2edge = vec![];
        let mut new_normals = vec![];
        let mut new_length = vec![];

        let mut skipped_edges = 0;
        for (i, edge) in self.edges.iter().enumerate() {
            match self.edge2elem[i].1 {
                Elem(_) => {
                    new_edges.push(*edge);
                    new_edge2elem.push(self.edge2elem[i].clone());
                    new_edge2edge.push(self.edge2edge[i].clone());
                    new_normals.push(self.normal[i]);
                    new_length.push(self.length[i]);
                }
                _ => {
                    //println!("skip edge {}", i);
                    skipped_edges += 1;
                }
            }
        }
        println!("skipped edges = {}", skipped_edges);
        self.edges = new_edges;
        self.edge2elem = new_edge2elem;
        self.edge2edge = new_edge2edge;
        self.normal = new_normals;
        self.length = new_length;
        self.nbedges = new_nbedges;

        // finally, update elem2elem
        for ((i1, i2), (iloc1, iloc2)) in self.edge2elem.iter().zip(self.edge2edge.iter()) {
            match i2 {
                Elem(i2) => {
                    self.elem2elem[*i1][*iloc1] = Elem(*i2);
                    if let LocEdge(iloc) = iloc2 {
                        self.elem2elem[*i2][*iloc] = Elem(*i1);
                    } else {
                        panic!("unexpected boundary type");
                    }
                }
                _ => {
                    panic!("unexpected boundary type");
                }
            }
        }

        assert_eq!(self.nbedges, self.edges.len());

        Ok(())
    }

    // save the mesh in a gmsh file format legacy 2
    pub fn save_gmsh2(&self, filename: &str, toplot: Option<Vec<f64>>) {
        let mut file = std::fs::File::create(filename).unwrap();
        let mut s = String::new();
        s.push_str("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");
        s.push_str("$Nodes\n");
        let nbnodes = self.vertices.len();
        s.push_str(&format!("{}\n", nbnodes));
        for (i, (x, y, _)) in self.vertices.iter().enumerate() {
            s.push_str(&format!("{} {} {} 0\n", i + 1, x, y));
        }
        s.push_str("$EndNodes\n");

        s.push_str("$Elements\n");
        let nbtri = self.elems.len();
        let nblines = self.edges.len();
        let nbelems = nbtri + nblines;
        // header
        s.push_str(&format!("{}\n", nbelems));
        // block of lines
        for (i, edge) in self.edges.iter().enumerate() {
            s.push_str(&format!("{} 8 0 ", i + 1 + nbtri));
            for node in edge {
                s.push_str(&format!("{} ", node + 1));
            }
            s.push('\n');
        }
        // block of triangles
        for (i, elem) in self.elems.iter().enumerate() {
            s.push_str(&format!("{} 9 0 ", i + 1));
            for node in elem {
                s.push_str(&format!("{} ", node + 1));
            }
            s.push('\n');
        }

        s.push_str("$EndElements\n");
        // save values
        // fn toplot(x: f64, y: f64) -> f64 {
        //     x * x + y * y
        // }

        // s.push_str(format!("$NodeData\n1\n\"field\"\n1\n0.0\n3\n0\n1\n{}\n", nbnodes).as_str());
        // for (i, (x, y, _)) in self.vertices.iter().enumerate() {
        //     s.push_str(&format!("{} {}\n", i + 1, toplot(*x, *y)));
        // }
        // s.push_str("$EndNodeData\n");

        match toplot {
            Some(toplot) => {
                s.push_str(
                    format!(
                        "$ElementNodeData\n1\n\"field\"\n1\n0.0\n3\n0\n1\n{}\n",
                        nbtri
                    )
                    .as_str(),
                );
                for (i, elem) in self.elems.iter().enumerate() {
                    s.push_str(&format!("{} 6 ", i + 1));
                    for node in elem {
                        s.push_str(&format!("{} ", toplot[i]));
                        //plot elem surface
                        //s.push_str(&format!("{} ", self.surface[i]));
                    }
                    s.push('\n');
                }
            }
            None => {}
        }

        file.write_all(s.as_bytes()).unwrap();
    }

    // check several properties of the mesh
    pub fn check(&self) {
        // check that the mesh is 2D
        // check that all elements have six nodes
        assert!(self.elems.iter().all(|x| x.len() == 6));
        // check that all the nodes are used
        let mut node_bis = vec![];
        for elem in &self.elems {
            for node in elem {
                node_bis.push(node);
            }
        }
        node_bis.sort();
        node_bis.dedup();
        assert_eq!(node_bis.len(), self.vertices.len());
        // check that edge2edge points to the correct local edges
        (self.edge2edge.iter())
            .zip(self.edge2elem.iter())
            .for_each(|(e2e, e2elem)| {
                let ie1 = e2elem.0;
                if let Elem(ie2) = e2elem.1 {
                    let iloc1 = e2e.0;
                    if let LocEdge(iloc2) = e2e.1 {
                        let a1 = self.elems[ie1][iloc1];
                        let b1 = self.elems[ie1][(iloc1 + 1) % 3];
                        let b2 = self.elems[ie2][iloc2];
                        let a2 = self.elems[ie2][(iloc2 + 1) % 3];
                        assert!(a1 == a2 && b1 == b2);
                    }
                }
            });
    }

    // write the mesh in a gmsh file
    // with a possible DG field to plot
    // fn plot_gmsh(&self, filename: &str, field: Option<Vec<f64>>) {
    //     todo!();
    // }
}

use crate::BoundaryType::Dirichlet;
use crate::BoundaryType::Elem;
use crate::BoundaryType::LocEdge;
use crate::BoundaryType::Neumann;
use std::collections::HashMap;

fn build_connectivity(
    elems: &Vec<Vec<usize>>,
) -> (
    Vec<[usize; 3]>,
    Vec<Vec<BoundaryType>>,
    Vec<(usize, BoundaryType)>,
    Vec<(usize, BoundaryType)>,
) {
    let mut edge_hash: HashMap<(usize, usize), usize> = HashMap::new();
    for (ie, elem) in elems.iter().enumerate() {
        for i in 0..3 {
            let i1 = elem[i];
            let i2 = elem[(i + 1) % 3];
            edge_hash.insert((i1, i2), ie);
        }
    }
    let mut elem2elem = vec![];
    for elem in elems {
        let mut elem2elem_elem = vec![];
        for i in 0..3 {
            let i1 = elem[i];
            let i2 = elem[(i + 1) % 3];
            let ie = edge_hash.get(&(i2, i1));
            match ie {
                Some(ie) => elem2elem_elem.push(BoundaryType::Elem(*ie)),
                None => elem2elem_elem.push(BoundaryType::Dirichlet),
            }
        }
        elem2elem.push(elem2elem_elem);
    }
    let mut edge2elem = vec![];
    let mut edges = vec![];
    let mut edge2edge = vec![];
    for (ie, elem) in elems.iter().enumerate() {
        for i in 0..3 {
            let i1 = elem[i];
            let i2 = elem[(i + 1) % 3];
            let i3 = elem[i + 3];
            // println!("{:?}", i3);
            // assert!(elem.len()==6);
            // assert!(i3==3 || i3==4 || i3==5);
            let ivois = elem2elem[ie][i].clone();
            match ivois {
                Elem(ie2) => {
                    if i1 < i2 {
                        edge2elem.push((ie, BoundaryType::Elem(ie2)));
                        edges.push([i1, i2, i3]);
                        // find local edge number in ie2: in ie2, the edge is i2->i1
                        // while in ie, the edge is i1->i2
                        let mut iloc = 0;
                        while (elems[ie2][iloc] != i2) || (elems[ie2][(iloc + 1) % 3] != i1) {
                            iloc += 1;
                        }
                        assert!(iloc < 3);
                        edge2edge.push((i, BoundaryType::LocEdge(iloc))); // left edge is i
                    }
                }
                Dirichlet => {
                    edge2elem.push((ie, BoundaryType::Dirichlet));
                    edges.push([i1, i2, i3]);
                    edge2edge.push((i, BoundaryType::Dirichlet));
                }
                Neumann => {
                    edge2elem.push((ie, BoundaryType::Neumann));
                    edges.push([i1, i2, i3]);
                }
                _ => {
                    panic!("Unexpected boundary type")
                }
            }
        }
    }
    (edges, elem2elem, edge2elem, edge2edge)
}

use nom::{
    //branch::alt,
    bytes::complete::{tag, take_until},
    //character::complete::{alpha1, char, none_of},
    combinator::map,
    error::{
        Error,
        ErrorKind::{self},
    },
    multi::fold_many_m_n,
    sequence::{preceded, terminated},
    IResult,
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
    let res = parse_line(input);
    match res {
        Ok((rest, line)) => {
            let v: Result<Vec<f64>, _> =
                line.split_whitespace().map(|x| x.parse::<f64>()).collect();
            match v {
                Ok(v) => Ok((rest, v)),
                Err(_) => Err(nom::Err::Error(Error::new(input, ErrorKind::Tag))),
            }
        }
        Err(e) => Err(e),
    }
}

fn parse_nodes_header(input: &str) -> IResult<&str, (usize, usize, usize, usize)> {
    map(parse_line_usizes, |v| {
        // let numblocks = v[0];
        // let numnodes = v[1];
        // let minnode = v[2];
        // let maxnode = v[3];
        (v[0], v[1], v[2], v[3])
    })(input)
}

fn parse_nodes_block_header(input: &str) -> IResult<&str, (usize, usize, usize, usize)> {
    map(parse_line_usizes, |v| {
        // let edim = v[0];
        // let etag = v[1];
        // let is_param = v[2];
        // let numnodes = v[3];
        (v[0], v[1], v[2], v[3])
    })(input)
}

fn parse_nodes_block(input: &str) -> IResult<&str, (Vec<usize>, Vec<(f64, f64, f64)>)> {
    let (rest, (_, _, _, nbnodes)) = parse_nodes_block_header(input)?;
    let (rest, nodenum) = fold_many_m_n(
        nbnodes,
        nbnodes,
        parse_line_usizes,
        Vec::<usize>::new,
        |mut acc, item| {
            acc.push(item[0]);
            acc
        },
    )(rest)?;
    let (rest, nodecoords) = fold_many_m_n(
        nbnodes,
        nbnodes,
        parse_line_f64s,
        Vec::<(f64, f64, f64)>::new,
        |mut acc, item| {
            acc.push((item[0], item[1], item[2]));
            acc
        },
    )(rest)?;
    Ok((rest, (nodenum, nodecoords)))
}

fn parse_nodes(input: &str) -> IResult<&str, (Vec<usize>, Vec<(f64, f64, f64)>)> {
    let (rest, _) = preceded(take_until("$Nodes\n"), tag("$Nodes\n"))(input)?;
    let (rest, (nbblocks, _, _, _)) = parse_nodes_header(rest)?;
    let (rest, (num, coord)) = fold_many_m_n(
        nbblocks,
        nbblocks,
        parse_nodes_block,
        || (Vec::<usize>::new(), Vec::<(f64, f64, f64)>::new()),
        |(mut accnum, mut acccoord), (num, coord)| {
            accnum.extend(num);
            acccoord.extend(coord);
            (accnum, acccoord)
        },
    )(rest)?;
    let (rest, _) = tag("$EndNodes\n")(rest)?;
    Ok((rest, (num, coord)))
}

fn parse_elems_header(input: &str) -> IResult<&str, (usize, usize, usize, usize)> {
    map(parse_line_usizes, |v| {
        // let numblocks = v[0];
        // let numelems = v[1];
        // let minelem = v[2];
        // let maxelem = v[3];
        (v[0], v[1], v[2], v[3])
    })(input)
}

fn parse_elems_block_header(input: &str) -> IResult<&str, (usize, usize, usize, usize)> {
    map(parse_line_usizes, |v| {
        // let edim = v[0];
        // let etag = v[1];
        // let elem_type = v[2];
        // let nbelems = v[3];
        (v[0], v[1], v[2], v[3])
    })(input)
}

fn parse_elems_block(input: &str) -> IResult<&str, Vec<Vec<usize>>> {
    let (rest, (_, _, elem_block_type, nbelems)) = parse_elems_block_header(input)?;
    let elem_type = 9; // only keep order 2 triangles
                       //println!("{:?}", elem_block_type);
    let (rest, elem) = fold_many_m_n(
        nbelems,
        nbelems,
        parse_line_usizes,
        || vec![],
        |mut acc, item| {
            if elem_block_type == elem_type {
                // remove first element of item
                // and substract 1 to all elements
                acc.push(item[1..].iter().map(|x| x - 1).collect());
            }
            acc
        },
    )(rest)?;
    Ok((rest, elem))
}

fn parse_elems(input: &str) -> IResult<&str, Vec<Vec<usize>>> {
    let (rest, _) = preceded(take_until("$Elements\n"), tag("$Elements\n"))(input)?;
    let (rest, (nbblocks, _, _, _)) = parse_elems_header(rest)?;
    let (rest, elem) = fold_many_m_n(
        nbblocks,
        nbblocks,
        parse_elems_block,
        || vec![],
        |mut acc, item| {
            acc.extend(item);
            acc
        },
    )(rest)?;
    let (rest, _) = tag("$EndElements\n")(rest)?;
    Ok((rest, elem))
}

fn parse_nodes_elems(input: &str) -> IResult<&str, (Vec<(f64, f64, f64)>, Vec<Vec<usize>>)> {
    let (rest, (_num, coord)) = parse_nodes(input)?;
    let (rest, elem) = parse_elems(rest)?;
    Ok((rest, (coord, elem)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mesh2d() {
        let mesh = Mesh2D::new("geo/square.msh");
        println!("{:?}", mesh);
        assert_eq!(mesh.vertices.len(), 13);
        assert_eq!(mesh.elems.len(), 4);
    }

    #[test]
    fn test_center() {
        let mesh = Mesh2D::new("geo/square.msh");
        let center = mesh.get_center(0);
        println!("{:?}", center);
        assert!((center.0 - 0.5).abs() + (center.1 - 0.5 / 3.).abs() < 1e-12);
    }

    #[test]
    fn test_surf() {
        let mesh = Mesh2D::new("geo/square.msh");
        let surf = mesh.get_surf(0);
        println!("{:?}", surf);
        assert!((surf - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_length() {
        let mesh = Mesh2D::new("geo/square.msh");
        let length = mesh.get_length(0, 0);
        println!("{:?}", length);
        assert!((length - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_perimeter() {
        let mesh = Mesh2D::new("geo/square.msh");
        let perimeter = mesh.get_perimeter(0);
        println!("{:?}", perimeter);
        assert!((perimeter - 1. - (2f64).sqrt()) < 1e-12);
    }

    #[test]
    fn test_save_gmsh() {
        let mesh = Mesh2D::new("geo/square4.msh");
        let nbel = mesh.get_nbelems();
        let toplot = (0..nbel).map(|x| x as f64).collect();
        mesh.save_gmsh2("geo/square_test.msh", Some(toplot));
        mesh.save_gmsh2("geo/void_square_test.msh", None);
    }

    #[test]
    fn test_make_periodic() {
        let mut mesh = Mesh2D::new("geo/square4.msh");
        mesh.make_periodic().unwrap();
        println!("{:?}", mesh);
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

    #[test]
    fn test_parse_nodes_block_header() {
        let line = "1 1 0 4\n";
        let (rest, parsed) = parse_nodes_block_header(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(parsed, (1, 1, 0, 4));
    }

    #[test]
    fn test_parse_nodes_block() {
        let line = r#"2 1 0 2
                      9  
                      10
                     0 0 0
                     1 0 0
"#;
        let (rest, (num, coord)) = parse_nodes_block(line).unwrap();
        println!("{:?}", num);
        println!("{:?}", coord);
        println!("{:?}", rest);
        assert_eq!(rest, "");
        assert_eq!(num, vec![9, 10]);
        assert_eq!(coord, vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);
    }

    #[test]
    // to run this single test: cargo test  tests::test_parse_nodes -- --exact --nocapture
    fn test_parse_nodes() {
        let line = r#"bibi
$Nodes
2 0 2 0
1 1 0 2
9
10
0 0 0
1 0 0
1 1 0 1
11
1 1 0
$EndNodes
boubou
"#;
        let (rest, (num, coord)) = parse_nodes(line).unwrap();
        println!("{:?}", num);
        println!("{:?}", coord);
        println!("{:?}", rest);
        assert_eq!(rest, "boubou\n");
        assert_eq!(num, vec![9, 10, 11]);
        assert_eq!(
            coord,
            vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        );
    }

    #[test]
    fn test_parse_elems_header() {
        let line = "10 0 3 0\n";
        let (rest, parsed) = parse_elems_header(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(parsed, (10, 0, 3, 0));
    }

    #[test]
    fn test_parse_elems_block_header() {
        let line = "1 1 0 4\n";
        let (rest, parsed) = parse_elems_block_header(line).unwrap();
        assert_eq!(rest, "");
        assert_eq!(parsed, (1, 1, 0, 4));
    }

    #[test]
    fn test_parse_elems_block() {
        let line = r#"2 1 9 2
        9 1 2 9 5 10 11 
        10 4 1 9 8 11 12 
"#;
        let (rest, elem) = parse_elems_block(line).unwrap();
        println!("{:?}", elem);
        println!("{:?}", rest);
        assert_eq!(rest, "");
        assert_eq!(
            elem,
            vec![vec![0, 1, 8, 4, 9, 10], vec![3, 0, 8, 7, 10, 11]]
        );
    }

    #[test]
    // to run this single test: cargo test  tests::test_parse_elems -- --exact --nocapture
    fn test_parse_elems() {
        let line = r#"bibi
$Elements
2 0 2 0
1 1 9 1
9 1 2 9 5 10 11
1 1 9 1
10 4 1 9 8 11 12
$EndElements
boubou
"#;
        let (rest, elem) = parse_elems(line).unwrap();
        println!("{:?}", elem);
        println!("{:?}", rest);
        assert_eq!(rest, "boubou\n");
        assert_eq!(
            elem,
            vec![vec![0, 1, 8, 4, 9, 10], vec![3, 0, 8, 7, 10, 11]]
        );
    }
}
