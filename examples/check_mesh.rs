use rsmsh::Mesh2D;

fn main() {
    let mesh = Mesh2D::new("geo/square.msh");
    println!("Mesh {:?}", mesh);
    println!("Mesh is valid: {}", mesh.check());
}