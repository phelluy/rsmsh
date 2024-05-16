use rsmsh::Mesh2D;

fn main() {
    let mesh = Mesh2D::new("geo/square4.msh");
    println!("Mesh {:?}", mesh);
    mesh.check();
}