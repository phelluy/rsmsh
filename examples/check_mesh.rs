use rsmsh::Mesh2D;

fn main() {
    let mesh = Mesh2D::new("geo/square.msh");
    println!("Mesh {:?}", mesh);
    mesh.check();
}